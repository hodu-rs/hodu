use crate::{
    backends::{
        be_hodu::storage::{HoduStorage, HoduStorageT},
        executor::{CompileOptions, ExecutionInputs, ExecutionOutputs, ExecutorT},
        op::{
            BinaryLogicalOp, BinaryOp, CastOp, CmpOp, CmpScalarOp, MatrixOp, Op, UnaryLogicalOp, UnaryOp, UnaryScalarOp,
        },
        script::{
            ir::{NodeId, ScriptIR},
            Script,
        },
    },
    compat::*,
    error::{HoduError, HoduResult},
    scalar::Scalar,
    tensor::{from_storage, Tensor, TensorId},
    types::{device::Device, dtype::DType, layout::Layout},
};

#[derive(Debug)]
pub struct HoduExecutor {
    current_device: Device,
}

#[derive(Debug)]
pub struct HoduCompiledScript {
    execution_plan: Vec<CompiledNode>,
    input_mapping: HashMap<String, TensorId>,
    output_mapping: HashMap<String, TensorId>,
    constant_storage: HashMap<TensorId, HoduStorage>,
    tensor_layouts: HashMap<TensorId, Layout>,
    tensor_dtypes: HashMap<TensorId, DType>,
}

#[derive(Debug)]
pub struct CompiledNode {
    id: NodeId,
    operation: Op,
    #[allow(dead_code)] // Will be used in CUDA/Metal implementations
    input_tensors: Vec<TensorId>,
    output_tensors: Vec<TensorId>,
    input_layouts: Vec<Layout>,
    #[allow(dead_code)] // Will be used in CUDA/Metal implementations
    output_layouts: Vec<Layout>,
}

impl HoduExecutor {
    pub fn new(device: Device) -> Self {
        Self { current_device: device }
    }

    fn convert_script_ir_to_compiled_nodes(&self, script_ir: &ScriptIR) -> HoduResult<Vec<CompiledNode>> {
        let mut compiled_nodes = Vec::new();

        for node in &script_ir.graph.topology.nodes {
            let compiled_node = CompiledNode {
                id: node.id,
                operation: node.operation.clone(),
                input_tensors: node.input_tensors.clone(),
                output_tensors: node.output_tensors.clone(),
                input_layouts: node.input_layouts.clone(),
                output_layouts: node.output_layouts.clone(),
            };
            compiled_nodes.push(compiled_node);
        }

        compiled_nodes.sort_by_key(|node| {
            script_ir
                .graph
                .execution_plan
                .execution_order
                .iter()
                .position(|&id| id == node.id)
                .unwrap_or(usize::MAX)
        });

        Ok(compiled_nodes)
    }

    fn collect_tensor_layouts(&self, script_ir: &ScriptIR) -> HashMap<TensorId, Layout> {
        let mut tensor_layouts = HashMap::new();

        for node in &script_ir.graph.topology.nodes {
            for (layout, &tensor_id) in node.input_layouts.iter().zip(&node.input_tensors) {
                tensor_layouts.insert(tensor_id, layout.clone());
            }
            for (layout, &tensor_id) in node.output_layouts.iter().zip(&node.output_tensors) {
                tensor_layouts.insert(tensor_id, layout.clone());
            }
        }

        for input in &script_ir.graph.metadata.inputs {
            if let Some(tensor_info) = script_ir.graph.metadata.tensor_info.get(&input.tensor_id) {
                if let Some(ref shape) = tensor_info.shape {
                    let shape_usize: Vec<usize> = shape.iter().map(|s| s.unwrap_or(1)).collect();
                    tensor_layouts.insert(input.tensor_id, Layout::from_shape(&shape_usize));
                }
            }
        }

        for output in &script_ir.graph.metadata.outputs {
            if let Some(tensor_info) = script_ir.graph.metadata.tensor_info.get(&output.tensor_id) {
                if let Some(ref shape) = tensor_info.shape {
                    let shape_usize: Vec<usize> = shape.iter().map(|s| s.unwrap_or(1)).collect();
                    tensor_layouts.insert(output.tensor_id, Layout::from_shape(&shape_usize));
                }
            }
        }

        tensor_layouts
    }

    fn collect_tensor_dtypes(&self, script_ir: &ScriptIR) -> HashMap<TensorId, DType> {
        let mut tensor_dtypes = HashMap::new();

        for (&tensor_id, tensor_info) in &script_ir.graph.metadata.tensor_info {
            if let Some(dtype) = tensor_info.dtype {
                tensor_dtypes.insert(tensor_id, dtype);
            }
        }

        tensor_dtypes
    }

    fn prepare_constant_storage(&self, script_ir: &ScriptIR) -> HoduResult<HashMap<TensorId, HoduStorage>> {
        let mut constant_storage = HashMap::new();

        for (tensor_id, constant_node) in &script_ir.graph.metadata.constants {
            match self.current_device {
                Device::CPU => {
                    let storage = self.convert_constant_to_cpu_storage(constant_node)?;
                    constant_storage.insert(*tensor_id, HoduStorage::CPU(storage));
                },
                Device::CUDA(_) => {
                    // TODO: Convert to CUDA storage
                    return Err(HoduError::InternalError(
                        "CUDA constant conversion not implemented".to_string(),
                    ));
                },
                Device::METAL(_) => {
                    // TODO: Convert to Metal storage
                    return Err(HoduError::InternalError(
                        "Metal constant conversion not implemented".to_string(),
                    ));
                },
            }
        }

        Ok(constant_storage)
    }

    fn convert_constant_to_cpu_storage(
        &self,
        constant: &crate::backends::script::ir::ConstantNode,
    ) -> HoduResult<crate::backends::be_hodu::cpu::storage::CpuStorage> {
        use crate::backends::be_hodu::cpu::storage::CpuStorage;
        use crate::backends::script::ir::CompressionType;
        use crate::types::dtype::DType;
        use float8::{F8E4M3, F8E5M2};
        use half::{bf16, f16};

        // Handle decompression if needed
        let data = match &constant.compression {
            #[cfg(feature = "std")]
            Some(CompressionType::Gzip) => {
                use std::io::Read;
                let mut decoder = flate2::read::GzDecoder::new(&constant.data[..]);
                let mut decompressed = Vec::new();
                decoder
                    .read_to_end(&mut decompressed)
                    .map_err(|e| HoduError::DecompressionError(e.to_string()))?;
                decompressed
            },
            Some(CompressionType::Zstd) => {
                return Err(HoduError::InternalError(
                    "Zstd decompression not implemented".to_string(),
                ));
            },
            _ => constant.data.clone(),
        };

        let elem_count = constant.shape.iter().product::<usize>();

        let cpu_storage = match constant.dtype {
            DType::BOOL => {
                let values: Vec<bool> = data
                    .chunks_exact(1)
                    .take(elem_count)
                    .map(|chunk| chunk[0] != 0)
                    .collect();
                CpuStorage::BOOL(values)
            },
            DType::F8E4M3 => {
                let values: Vec<F8E4M3> = data
                    .chunks_exact(1)
                    .take(elem_count)
                    .map(|chunk| F8E4M3::from_bits(chunk[0]))
                    .collect();
                CpuStorage::F8E4M3(values)
            },
            DType::F8E5M2 => {
                let values: Vec<F8E5M2> = data
                    .chunks_exact(1)
                    .take(elem_count)
                    .map(|chunk| F8E5M2::from_bits(chunk[0]))
                    .collect();
                CpuStorage::F8E5M2(values)
            },
            DType::BF16 => {
                let values: Vec<bf16> = data
                    .chunks_exact(2)
                    .take(elem_count)
                    .map(|chunk| bf16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                CpuStorage::BF16(values)
            },
            DType::F16 => {
                let values: Vec<f16> = data
                    .chunks_exact(2)
                    .take(elem_count)
                    .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                CpuStorage::F16(values)
            },
            DType::F32 => {
                let values: Vec<f32> = data
                    .chunks_exact(4)
                    .take(elem_count)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                CpuStorage::F32(values)
            },
            DType::F64 => {
                let values: Vec<f64> = data
                    .chunks_exact(8)
                    .take(elem_count)
                    .map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                        ])
                    })
                    .collect();
                CpuStorage::F64(values)
            },
            DType::U8 => {
                let values: Vec<u8> = data.into_iter().take(elem_count).collect();
                CpuStorage::U8(values)
            },
            DType::U16 => {
                let values: Vec<u16> = data
                    .chunks_exact(2)
                    .take(elem_count)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                CpuStorage::U16(values)
            },
            DType::U32 => {
                let values: Vec<u32> = data
                    .chunks_exact(4)
                    .take(elem_count)
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                CpuStorage::U32(values)
            },
            DType::U64 => {
                let values: Vec<u64> = data
                    .chunks_exact(8)
                    .take(elem_count)
                    .map(|chunk| {
                        u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                        ])
                    })
                    .collect();
                CpuStorage::U64(values)
            },
            DType::I8 => {
                let values: Vec<i8> = data.into_iter().take(elem_count).map(|b| b as i8).collect();
                CpuStorage::I8(values)
            },
            DType::I16 => {
                let values: Vec<i16> = data
                    .chunks_exact(2)
                    .take(elem_count)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                CpuStorage::I16(values)
            },
            DType::I32 => {
                let values: Vec<i32> = data
                    .chunks_exact(4)
                    .take(elem_count)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                CpuStorage::I32(values)
            },
            DType::I64 => {
                let values: Vec<i64> = data
                    .chunks_exact(8)
                    .take(elem_count)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                        ])
                    })
                    .collect();
                CpuStorage::I64(values)
            },
        };

        Ok(cpu_storage)
    }

    fn validate_inputs(&self, compiled: &HoduCompiledScript, inputs: &ExecutionInputs<'_>) -> HoduResult<()> {
        for input_name in compiled.input_mapping.keys() {
            if !inputs.contains_key(input_name.as_str()) {
                return Err(HoduError::InternalError(format!(
                    "Missing required input: {input_name}"
                )));
            }
        }

        for input_name in inputs.keys() {
            if !compiled.input_mapping.contains_key(*input_name) {
                return Err(HoduError::InternalError(format!("Unexpected input: {input_name}")));
            }
        }

        Ok(())
    }

    fn tensor_to_storage(&self, tensor: &Tensor) -> HoduResult<HoduStorage> {
        tensor.with_storage(|storage| match (&self.current_device, storage) {
            (Device::CPU, storage) => storage.to_cpu_storage().map(HoduStorage::CPU),
            (Device::CUDA(_), _) => Err(HoduError::InternalError(
                "CUDA tensor conversion not implemented".to_string(),
            )),
            (Device::METAL(_), _) => Err(HoduError::InternalError(
                "Metal tensor conversion not implemented".to_string(),
            )),
        })
    }

    fn execute_node(
        &self,
        compiled_node: &CompiledNode,
        tensor_storage: &HashMap<TensorId, HoduStorage>,
        compiled: &HoduCompiledScript,
    ) -> HoduResult<HoduStorage> {
        match &compiled_node.operation {
            Op::Binary(binary_op, lhs_id, rhs_id) => {
                let lhs_storage = tensor_storage
                    .get(lhs_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {lhs_id:?} not found")))?;
                let rhs_storage = tensor_storage
                    .get(rhs_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {rhs_id:?} not found")))?;
                let lhs_layout = &compiled_node.input_layouts[0];
                let rhs_layout = &compiled_node.input_layouts[1];

                self.execute_binary_op(*binary_op, lhs_storage, rhs_storage, lhs_layout, rhs_layout)
            },
            Op::BinaryLogical(binary_logical_op, lhs_id, rhs_id) => {
                let lhs_storage = tensor_storage
                    .get(lhs_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {lhs_id:?} not found")))?;
                let rhs_storage = tensor_storage
                    .get(rhs_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {rhs_id:?} not found")))?;
                let lhs_layout = &compiled_node.input_layouts[0];
                let rhs_layout = &compiled_node.input_layouts[1];

                self.execute_binary_logical_op(*binary_logical_op, lhs_storage, rhs_storage, lhs_layout, rhs_layout)
            },
            Op::Cmp(cmp_op, lhs_id, rhs_id) => {
                let lhs_storage = tensor_storage
                    .get(lhs_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {lhs_id:?} not found")))?;
                let rhs_storage = tensor_storage
                    .get(rhs_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {rhs_id:?} not found")))?;
                let lhs_layout = &compiled_node.input_layouts[0];
                let rhs_layout = &compiled_node.input_layouts[1];

                self.execute_cmp_op(*cmp_op, lhs_storage, rhs_storage, lhs_layout, rhs_layout)
            },
            Op::CmpScalar(cmp_scalar_op, tensor_id, scalar) => {
                let tensor_storage = tensor_storage
                    .get(tensor_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {tensor_id:?} not found")))?;
                let layout = &compiled_node.input_layouts[0];

                self.execute_cmp_scalar_op(*cmp_scalar_op, tensor_storage, layout, *scalar)
            },
            Op::Unary(unary_op, tensor_id) => {
                let input_storage = tensor_storage
                    .get(tensor_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {tensor_id:?} not found")))?;
                let layout = &compiled_node.input_layouts[0];

                self.execute_unary_op(*unary_op, input_storage, layout)
            },
            Op::UnaryLogical(unary_logical_op, tensor_id) => {
                let input_storage = tensor_storage
                    .get(tensor_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {tensor_id:?} not found")))?;
                let layout = &compiled_node.input_layouts[0];

                self.execute_unary_logical_op(*unary_logical_op, input_storage, layout)
            },
            Op::UnaryScalar(unary_scalar_op, tensor_id, scalar) => {
                let input_storage = tensor_storage
                    .get(tensor_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {tensor_id:?} not found")))?;
                let layout = &compiled_node.input_layouts[0];

                self.execute_unary_scalar_op(*unary_scalar_op, input_storage, layout, *scalar)
            },
            Op::Matrix(matrix_op, lhs_id, rhs_id) => {
                let lhs_storage = tensor_storage
                    .get(lhs_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {lhs_id:?} not found")))?;
                let rhs_storage = tensor_storage
                    .get(rhs_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {rhs_id:?} not found")))?;
                let lhs_layout = &compiled_node.input_layouts[0];
                let rhs_layout = &compiled_node.input_layouts[1];

                self.execute_matrix_op(*matrix_op, lhs_storage, rhs_storage, lhs_layout, rhs_layout)
            },
            Op::Cast(cast_op, tensor_id) => {
                let input_storage = tensor_storage
                    .get(tensor_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {tensor_id:?} not found")))?;

                self.execute_cast_op(*cast_op, input_storage, compiled_node, compiled)
            },
            _ => Err(HoduError::InternalError(format!(
                "Operation {:?} not implemented yet",
                compiled_node.operation
            ))),
        }
    }

    fn execute_binary_op(
        &self,
        binary_op: BinaryOp,
        lhs_storage: &HoduStorage,
        rhs_storage: &HoduStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> HoduResult<HoduStorage> {
        use crate::backends::op::*;

        match binary_op {
            BinaryOp::Add => lhs_storage.binary_impl::<Add>(rhs_storage, lhs_layout, rhs_layout),
            BinaryOp::Sub => lhs_storage.binary_impl::<Sub>(rhs_storage, lhs_layout, rhs_layout),
            BinaryOp::Mul => lhs_storage.binary_impl::<Mul>(rhs_storage, lhs_layout, rhs_layout),
            BinaryOp::Div => lhs_storage.binary_impl::<Div>(rhs_storage, lhs_layout, rhs_layout),
            BinaryOp::Pow => lhs_storage.binary_impl::<Pow>(rhs_storage, lhs_layout, rhs_layout),
            BinaryOp::Maximum => lhs_storage.binary_impl::<Maximum>(rhs_storage, lhs_layout, rhs_layout),
            BinaryOp::Minimum => lhs_storage.binary_impl::<Minimum>(rhs_storage, lhs_layout, rhs_layout),
        }
    }

    fn execute_binary_logical_op(
        &self,
        binary_logical_op: BinaryLogicalOp,
        lhs_storage: &HoduStorage,
        rhs_storage: &HoduStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> HoduResult<HoduStorage> {
        use crate::backends::op::*;

        match binary_logical_op {
            BinaryLogicalOp::LogicalAnd => {
                lhs_storage.binary_logical_impl::<LogicalAnd>(rhs_storage, lhs_layout, rhs_layout)
            },
            BinaryLogicalOp::LogicalOr => {
                lhs_storage.binary_logical_impl::<LogicalOr>(rhs_storage, lhs_layout, rhs_layout)
            },
            BinaryLogicalOp::LogicalXor => {
                lhs_storage.binary_logical_impl::<LogicalXor>(rhs_storage, lhs_layout, rhs_layout)
            },
        }
    }

    fn execute_cmp_op(
        &self,
        cmp_op: CmpOp,
        lhs_storage: &HoduStorage,
        rhs_storage: &HoduStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> HoduResult<HoduStorage> {
        use crate::backends::op::*;

        match cmp_op {
            CmpOp::Eq => lhs_storage.cmp_impl::<Eq>(rhs_storage, lhs_layout, rhs_layout),
            CmpOp::Ne => lhs_storage.cmp_impl::<Ne>(rhs_storage, lhs_layout, rhs_layout),
            CmpOp::Lt => lhs_storage.cmp_impl::<Lt>(rhs_storage, lhs_layout, rhs_layout),
            CmpOp::Le => lhs_storage.cmp_impl::<Le>(rhs_storage, lhs_layout, rhs_layout),
            CmpOp::Gt => lhs_storage.cmp_impl::<Gt>(rhs_storage, lhs_layout, rhs_layout),
            CmpOp::Ge => lhs_storage.cmp_impl::<Ge>(rhs_storage, lhs_layout, rhs_layout),
        }
    }

    fn execute_cmp_scalar_op(
        &self,
        cmp_scalar_op: CmpScalarOp,
        tensor_storage: &HoduStorage,
        layout: &Layout,
        scalar: Scalar,
    ) -> HoduResult<HoduStorage> {
        use crate::backends::op::*;

        match cmp_scalar_op {
            CmpScalarOp::EqScalar => tensor_storage.cmp_scalar_impl::<EqScalar>(layout, scalar),
            CmpScalarOp::NeScalar => tensor_storage.cmp_scalar_impl::<NeScalar>(layout, scalar),
            CmpScalarOp::LtScalar => tensor_storage.cmp_scalar_impl::<LtScalar>(layout, scalar),
            CmpScalarOp::LeScalar => tensor_storage.cmp_scalar_impl::<LeScalar>(layout, scalar),
            CmpScalarOp::GtScalar => tensor_storage.cmp_scalar_impl::<GtScalar>(layout, scalar),
            CmpScalarOp::GeScalar => tensor_storage.cmp_scalar_impl::<GeScalar>(layout, scalar),
        }
    }

    fn execute_unary_op(
        &self,
        unary_op: UnaryOp,
        input_storage: &HoduStorage,
        layout: &Layout,
    ) -> HoduResult<HoduStorage> {
        use crate::backends::op::*;

        match unary_op {
            UnaryOp::Neg => input_storage.unary_impl::<Neg>(layout),
            UnaryOp::Abs => input_storage.unary_impl::<Abs>(layout),
            UnaryOp::Sign => input_storage.unary_impl::<Sign>(layout),
            UnaryOp::Square => input_storage.unary_impl::<Square>(layout),
            UnaryOp::Relu => input_storage.unary_impl::<Relu>(layout),
            UnaryOp::Sigmoid => input_storage.unary_impl::<Sigmoid>(layout),
            UnaryOp::Tanh => input_storage.unary_impl::<Tanh>(layout),
            UnaryOp::Gelu => input_storage.unary_impl::<Gelu>(layout),
            UnaryOp::Sin => input_storage.unary_impl::<Sin>(layout),
            UnaryOp::Cos => input_storage.unary_impl::<Cos>(layout),
            UnaryOp::Tan => input_storage.unary_impl::<Tan>(layout),
            UnaryOp::Ln => input_storage.unary_impl::<Ln>(layout),
            UnaryOp::Log10 => input_storage.unary_impl::<Log10>(layout),
            UnaryOp::Log2 => input_storage.unary_impl::<Log2>(layout),
            UnaryOp::Exp => input_storage.unary_impl::<Exp>(layout),
            UnaryOp::Exp10 => input_storage.unary_impl::<Exp10>(layout),
            UnaryOp::Exp2 => input_storage.unary_impl::<Exp2>(layout),
            UnaryOp::Softplus => input_storage.unary_impl::<Softplus>(layout),
            UnaryOp::Recip => input_storage.unary_impl::<Recip>(layout),
            UnaryOp::Sqrt => input_storage.unary_impl::<Sqrt>(layout),
        }
    }

    fn execute_unary_logical_op(
        &self,
        unary_logical_op: UnaryLogicalOp,
        input_storage: &HoduStorage,
        layout: &Layout,
    ) -> HoduResult<HoduStorage> {
        use crate::backends::op::*;

        match unary_logical_op {
            UnaryLogicalOp::LogicalNot => input_storage.unary_logical_impl::<LogicalNot>(layout),
        }
    }

    fn execute_unary_scalar_op(
        &self,
        unary_scalar_op: UnaryScalarOp,
        input_storage: &HoduStorage,
        layout: &Layout,
        scalar: Scalar,
    ) -> HoduResult<HoduStorage> {
        use crate::backends::op::*;

        match unary_scalar_op {
            UnaryScalarOp::AddScalar => input_storage.unary_scalar_impl::<AddScalar>(layout, scalar),
            UnaryScalarOp::SubScalar => input_storage.unary_scalar_impl::<SubScalar>(layout, scalar),
            UnaryScalarOp::MulScalar => input_storage.unary_scalar_impl::<MulScalar>(layout, scalar),
            UnaryScalarOp::DivScalar => input_storage.unary_scalar_impl::<DivScalar>(layout, scalar),
            UnaryScalarOp::PowScalar => input_storage.unary_scalar_impl::<PowScalar>(layout, scalar),
            UnaryScalarOp::MaximumScalar => input_storage.unary_scalar_impl::<MaximumScalar>(layout, scalar),
            UnaryScalarOp::MinimumScalar => input_storage.unary_scalar_impl::<MinimumScalar>(layout, scalar),
            UnaryScalarOp::LeakyRelu => input_storage.unary_scalar_impl::<LeakyRelu>(layout, scalar),
            UnaryScalarOp::Elu => input_storage.unary_scalar_impl::<Elu>(layout, scalar),
        }
    }

    fn execute_matrix_op(
        &self,
        matrix_op: MatrixOp,
        lhs_storage: &HoduStorage,
        rhs_storage: &HoduStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> HoduResult<HoduStorage> {
        match matrix_op {
            MatrixOp::Matmul => lhs_storage.matmul(rhs_storage, lhs_layout, rhs_layout),
        }
    }

    fn execute_cast_op(
        &self,
        cast_op: CastOp,
        input_storage: &HoduStorage,
        compiled_node: &CompiledNode,
        compiled: &HoduCompiledScript,
    ) -> HoduResult<HoduStorage> {
        match cast_op {
            CastOp::ToDType => {
                // Get the output tensor ID to find its dtype
                let output_tensor_id = compiled_node
                    .output_tensors
                    .first()
                    .ok_or_else(|| HoduError::InternalError("Cast operation has no output tensor".to_string()))?;

                // Get target dtype from compiled tensor_dtypes
                let target_dtype = compiled
                    .tensor_dtypes
                    .get(output_tensor_id)
                    .copied()
                    .unwrap_or(DType::F32); // Default to F32 if no dtype specified

                match input_storage {
                    HoduStorage::CPU(cpu_storage) => {
                        let converted_storage = cpu_storage.to_dtype(target_dtype)?;
                        Ok(HoduStorage::CPU(converted_storage))
                    },
                }
            },
        }
    }
}

impl ExecutorT for HoduExecutor {
    type CompiledScript = HoduCompiledScript;

    fn backend_name(&self) -> &'static str {
        "hodu"
    }

    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::CPU]
    }

    fn current_device(&self) -> Device {
        self.current_device
    }

    fn compile(&mut self, script: &Script, _options: CompileOptions) -> HoduResult<Self::CompiledScript> {
        let script_ir = script
            .get_ir()
            .ok_or_else(|| HoduError::ScriptValidationFailed("Script has no IR".to_string()))?;
        script_ir.validate().map_err(HoduError::ScriptValidationFailed)?;

        let execution_plan = self.convert_script_ir_to_compiled_nodes(script_ir)?;

        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();

        for input in &script_ir.graph.metadata.inputs {
            input_mapping.insert(input.name.clone(), input.tensor_id);
        }
        for output in &script_ir.graph.metadata.outputs {
            output_mapping.insert(output.name.clone(), output.tensor_id);
        }
        let constant_storage = self.prepare_constant_storage(script_ir)?;
        let tensor_layouts = self.collect_tensor_layouts(script_ir);
        let tensor_dtypes = self.collect_tensor_dtypes(script_ir);

        Ok(HoduCompiledScript {
            execution_plan,
            input_mapping,
            output_mapping,
            constant_storage,
            tensor_layouts,
            tensor_dtypes,
        })
    }

    fn execute(&self, compiled: &Self::CompiledScript, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
        self.validate_inputs(compiled, &inputs)?;

        let mut tensor_storage: HashMap<TensorId, HoduStorage> = HashMap::new();

        for (&tensor_id, storage) in &compiled.constant_storage {
            let new_storage = match storage {
                HoduStorage::CPU(cpu_storage) => HoduStorage::CPU(cpu_storage.clone()),
            };
            tensor_storage.insert(tensor_id, new_storage);
        }

        for (input_name, input_tensor) in &inputs {
            if let Some(&tensor_id) = compiled.input_mapping.get(*input_name) {
                let storage = self.tensor_to_storage(input_tensor)?;
                tensor_storage.insert(tensor_id, storage);
            }
        }

        for compiled_node in &compiled.execution_plan {
            let result_storage = self.execute_node(compiled_node, &tensor_storage, compiled)?;

            for &output_tensor_id in &compiled_node.output_tensors {
                let cloned_result = match &result_storage {
                    HoduStorage::CPU(cpu_storage) => HoduStorage::CPU(cpu_storage.clone()),
                };
                tensor_storage.insert(output_tensor_id, cloned_result);
            }
        }

        let mut outputs = HashMap::new();
        for (output_name, &tensor_id) in &compiled.output_mapping {
            if let Some(storage) = tensor_storage.get(&tensor_id) {
                if let Some(layout) = compiled.tensor_layouts.get(&tensor_id) {
                    let cloned_storage = match storage {
                        HoduStorage::CPU(cpu_storage) => HoduStorage::CPU(cpu_storage.clone()),
                    };
                    let output_tensor = from_storage(cloned_storage, layout.clone(), false);
                    outputs.insert(output_name.clone(), output_tensor);
                } else {
                    return Err(HoduError::InternalError(format!(
                        "Layout not found for output tensor {tensor_id:?}"
                    )));
                }
            } else {
                return Err(HoduError::InternalError(format!(
                    "Storage not found for output tensor {tensor_id:?}"
                )));
            }
        }

        Ok(outputs)
    }

    fn cleanup(&mut self) -> HoduResult<()> {
        // Nothing to cleanup for now
        Ok(())
    }
}
