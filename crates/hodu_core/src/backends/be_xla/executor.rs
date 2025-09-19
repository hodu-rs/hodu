use crate::{
    backends::{
        executor::{CompileOptions, ExecutionInputs, ExecutionOutputs, ExecutorT},
        op::Op,
        script::{ir::ScriptIR, Script},
    },
    compat::*,
    error::{HoduError, HoduResult},
    tensor::{from_storage, Tensor, TensorId},
    types::{device::Device, dtype::DType, layout::Layout},
};
use std::collections::HashMap;
use std::sync::Arc;
use xla::{ElementType, Literal, PjRtClient, PjRtLoadedExecutable, XlaBuilder, XlaOp};

// Thread-safe wrapper for PjRtLoadedExecutable
// SAFETY: XLA's PjRtLoadedExecutable is thread-safe in practice,
// but the Rust bindings don't expose Send/Sync markers
struct ThreadSafeExecutable(PjRtLoadedExecutable);

unsafe impl Send for ThreadSafeExecutable {}
unsafe impl Sync for ThreadSafeExecutable {}

impl std::ops::Deref for ThreadSafeExecutable {
    type Target = PjRtLoadedExecutable;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Thread-safe wrapper for PjRtClient
// SAFETY: XLA's PjRtClient is thread-safe in practice,
// but the Rust bindings don't expose Send/Sync markers
struct ThreadSafeClient(PjRtClient);

unsafe impl Send for ThreadSafeClient {}
unsafe impl Sync for ThreadSafeClient {}

impl std::ops::Deref for ThreadSafeClient {
    type Target = PjRtClient;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Helper function to convert XLA errors to HoduError
fn xla_error_to_hodu_error(err: xla::Error) -> HoduError {
    HoduError::InternalError(format!("XLA error: {:?}", err))
}

pub struct XlaExecutor {
    client: Arc<ThreadSafeClient>,
    device: Device,
}

pub struct XlaCompiledScript {
    executable: Arc<ThreadSafeExecutable>,
    input_mapping: HashMap<String, TensorId>,
    output_mapping: HashMap<String, TensorId>,
    tensor_layouts: HashMap<TensorId, Layout>,
    tensor_dtypes: HashMap<TensorId, DType>,
}

impl XlaExecutor {
    /// Convert DType to XLA ElementType
    fn dtype_to_element_type(dtype: DType) -> HoduResult<ElementType> {
        match dtype {
            DType::F32 => Ok(ElementType::F32),
            DType::F64 => Ok(ElementType::F64),
            DType::I32 => Ok(ElementType::S32),
            DType::I64 => Ok(ElementType::S64),
            DType::U32 => Ok(ElementType::U32),
            DType::U64 => Ok(ElementType::U64),
            DType::BOOL => Ok(ElementType::Pred),
            DType::F16 => Ok(ElementType::F16),
            DType::BF16 => Ok(ElementType::Bf16),
            DType::I8 => Ok(ElementType::S8),
            DType::I16 => Ok(ElementType::S16),
            DType::U8 => Ok(ElementType::U8),
            DType::U16 => Ok(ElementType::U16),
            DType::F8E4M3 | DType::F8E5M2 => Err(HoduError::InternalError(format!(
                "XLA does not support {:?} dtype",
                dtype
            ))),
        }
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

    fn build_xla_computation(
        &self,
        compiled_script: &XlaCompiledScript,
        script_ir: &ScriptIR,
    ) -> HoduResult<Arc<ThreadSafeExecutable>> {
        let builder = XlaBuilder::new("xla_computation");
        let mut xla_ops = HashMap::new();

        // Create parameters for inputs in order (sort for consistent ordering)
        let mut input_names: Vec<_> = compiled_script.input_mapping.keys().cloned().collect();
        input_names.sort(); // Ensure consistent ordering

        for (i, input_name) in input_names.iter().enumerate() {
            if let Some(&tensor_id) = compiled_script.input_mapping.get(input_name) {
                if let Some(layout) = compiled_script.tensor_layouts.get(&tensor_id) {
                    let dtype = compiled_script
                        .tensor_dtypes
                        .get(&tensor_id)
                        .copied()
                        .unwrap_or(DType::F32);
                    let element_type = Self::dtype_to_element_type(dtype)?;
                    let dims: Vec<i64> = layout.get_shape().iter().map(|&d| d as i64).collect();

                    let param_name = format!("input_{}", i);
                    let xla_param = builder
                        .parameter(i as i64, element_type, &dims, &param_name)
                        .map_err(xla_error_to_hodu_error)?;

                    xla_ops.insert(tensor_id, xla_param);
                }
            }
        }

        // Process all nodes in execution order
        for &node_id in &script_ir.graph.execution_plan.execution_order {
            let node = &script_ir.graph.topology.nodes[node_id.0];

            // Get input operations
            let input_ops: Vec<XlaOp> = node
                .input_tensors
                .iter()
                .map(|&tensor_id| {
                    xla_ops.get(&tensor_id).cloned().ok_or_else(|| {
                        HoduError::InternalError(format!(
                            "Input tensor {:?} not found for node {:?}",
                            tensor_id, node_id
                        ))
                    })
                })
                .collect::<HoduResult<Vec<_>>>()?;

            // Execute the operation and store result
            let result_op = self.execute_xla_operation(&builder, &node.operation, &input_ops, compiled_script)?;

            // Store the result for output tensor
            if let Some(&output_tensor_id) = node.output_tensors.first() {
                xla_ops.insert(output_tensor_id, result_op);
            }
        }

        // Find the final output operation
        let output_names: Vec<_> = compiled_script.output_mapping.keys().cloned().collect();
        if output_names.len() != 1 {
            return Err(HoduError::InternalError(
                "XLA backend currently supports single output only".to_string(),
            ));
        }

        let output_tensor_id = compiled_script
            .output_mapping
            .get(&output_names[0])
            .ok_or_else(|| HoduError::InternalError("Output tensor mapping not found".to_string()))?;

        let final_op = xla_ops
            .get(output_tensor_id)
            .ok_or_else(|| HoduError::InternalError("Final output operation not found".to_string()))?;

        let computation = final_op.build().map_err(xla_error_to_hodu_error)?;
        let executable = self.client.compile(&computation).map_err(xla_error_to_hodu_error)?;
        Ok(Arc::new(ThreadSafeExecutable(executable)))
    }

    fn execute_xla_operation(
        &self,
        builder: &XlaBuilder,
        operation: &Op,
        input_ops: &[XlaOp],
        _compiled_script: &XlaCompiledScript,
    ) -> HoduResult<XlaOp> {
        match operation {
            // Binary Operations
            Op::Binary(op, _, _) => {
                if input_ops.len() != 2 {
                    return Err(HoduError::InternalError(
                        "Binary operation requires exactly 2 inputs".to_string(),
                    ));
                }
                let result = match op {
                    crate::backends::op::BinaryOp::Add => input_ops[0].add_(&input_ops[1]),
                    crate::backends::op::BinaryOp::Sub => input_ops[0].sub_(&input_ops[1]),
                    crate::backends::op::BinaryOp::Mul => input_ops[0].mul_(&input_ops[1]),
                    crate::backends::op::BinaryOp::Div => input_ops[0].div_(&input_ops[1]),
                    crate::backends::op::BinaryOp::Pow => input_ops[0].pow(&input_ops[1]),
                    crate::backends::op::BinaryOp::Maximum => input_ops[0].max(&input_ops[1]),
                    crate::backends::op::BinaryOp::Minimum => input_ops[0].min(&input_ops[1]),
                }
                .map_err(xla_error_to_hodu_error)?;
                Ok(result)
            },

            // Binary Logical Operations
            Op::BinaryLogical(op, _, _) => {
                if input_ops.len() != 2 {
                    return Err(HoduError::InternalError(
                        "Binary logical operation requires exactly 2 inputs".to_string(),
                    ));
                }
                let result = match op {
                    crate::backends::op::BinaryLogicalOp::LogicalAnd => input_ops[0].and(&input_ops[1]),
                    crate::backends::op::BinaryLogicalOp::LogicalOr => input_ops[0].or(&input_ops[1]),
                    crate::backends::op::BinaryLogicalOp::LogicalXor => input_ops[0].xor(&input_ops[1]),
                }
                .map_err(xla_error_to_hodu_error)?;
                Ok(result)
            },

            // Comparison Operations
            Op::Cmp(op, _, _) => {
                if input_ops.len() != 2 {
                    return Err(HoduError::InternalError(
                        "Comparison operation requires exactly 2 inputs".to_string(),
                    ));
                }
                let result = match op {
                    crate::backends::op::CmpOp::Eq => input_ops[0].eq(&input_ops[1]),
                    crate::backends::op::CmpOp::Ne => input_ops[0].ne(&input_ops[1]),
                    crate::backends::op::CmpOp::Lt => input_ops[0].lt(&input_ops[1]),
                    crate::backends::op::CmpOp::Le => input_ops[0].le(&input_ops[1]),
                    crate::backends::op::CmpOp::Gt => input_ops[0].gt(&input_ops[1]),
                    crate::backends::op::CmpOp::Ge => input_ops[0].ge(&input_ops[1]),
                }
                .map_err(xla_error_to_hodu_error)?;
                Ok(result)
            },

            // Unary Operations (simplified set)
            Op::Unary(op, _) => {
                if input_ops.len() != 1 {
                    return Err(HoduError::InternalError(
                        "Unary operation requires exactly 1 input".to_string(),
                    ));
                }
                let result = match op {
                    crate::backends::op::UnaryOp::Neg => input_ops[0].neg(),
                    crate::backends::op::UnaryOp::Abs => input_ops[0].abs(),
                    crate::backends::op::UnaryOp::Sign => input_ops[0].sign(),
                    crate::backends::op::UnaryOp::Square => input_ops[0].mul_(&input_ops[0]),
                    crate::backends::op::UnaryOp::Relu => {
                        let zero = builder.constant_r0(0.0f32).map_err(xla_error_to_hodu_error)?;
                        input_ops[0].max(&zero)
                    },
                    crate::backends::op::UnaryOp::Sigmoid => input_ops[0].logistic(), // Use XLA's builtin logistic
                    crate::backends::op::UnaryOp::Tanh => input_ops[0].tanh(),
                    crate::backends::op::UnaryOp::Sin => input_ops[0].sin(),
                    crate::backends::op::UnaryOp::Cos => input_ops[0].cos(),
                    crate::backends::op::UnaryOp::Ln => input_ops[0].log(),
                    crate::backends::op::UnaryOp::Exp => input_ops[0].exp(),
                    crate::backends::op::UnaryOp::Sqrt => input_ops[0].sqrt(),
                    _ => {
                        return Err(HoduError::InternalError(format!(
                            "Unary operation {:?} not yet implemented for XLA",
                            op
                        )))
                    },
                }
                .map_err(xla_error_to_hodu_error)?;
                Ok(result)
            },

            // Unary Logical Operations
            Op::UnaryLogical(op, _) => {
                if input_ops.len() != 1 {
                    return Err(HoduError::InternalError(
                        "Unary logical operation requires exactly 1 input".to_string(),
                    ));
                }
                let result = match op {
                    crate::backends::op::UnaryLogicalOp::LogicalNot => input_ops[0].not(),
                }
                .map_err(xla_error_to_hodu_error)?;
                Ok(result)
            },

            // Matrix Operations
            Op::Matrix(op, _, _) => {
                if input_ops.len() != 2 {
                    return Err(HoduError::InternalError(
                        "Matrix operation requires exactly 2 inputs".to_string(),
                    ));
                }
                let result = match op {
                    crate::backends::op::MatrixOp::Matmul => input_ops[0].dot(&input_ops[1]),
                }
                .map_err(xla_error_to_hodu_error)?;
                Ok(result)
            },

            // For now, return error for unsupported operations
            _ => Err(HoduError::InternalError(format!(
                "Operation {:?} not yet implemented for XLA backend",
                operation
            ))),
        }
    }

    /// Convert Hodu tensor to XLA literal
    fn tensor_to_literal(&self, tensor: &Tensor) -> HoduResult<Literal> {
        let layout = tensor.get_layout();
        let shape = layout.get_shape();
        let dims: Vec<usize> = shape.to_vec();

        tensor.with_storage(|storage| {
            let cpu_storage = storage
                .to_cpu_storage()
                .map_err(|e| HoduError::InternalError(format!("Failed to convert to CPU storage: {:?}", e)))?;

            use crate::backends::be_hodu::cpu::storage::CpuStorage;
            match cpu_storage {
                CpuStorage::BOOL(ref data) => {
                    // Convert bool to u8 as XLA doesn't support bool NativeType directly
                    let u8_data: Vec<u8> = data.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect();
                    if dims.is_empty() {
                        Ok(Literal::scalar(u8_data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(&u8_data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::Pred, &dims, &u8_data)
                                .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::BF16(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::Bf16, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<half::bf16>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::F16(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::F16, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<half::f16>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::F32(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::F32, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<f32>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::F64(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::F64, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<f64>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::U8(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::U8, &dims, data)
                                .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::U16(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::U16, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<u16>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::U32(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::U32, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<u32>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::U64(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::U64, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<u64>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::I8(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::S8, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<i8>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::I16(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::S16, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<i16>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::I32(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::S32, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<i32>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                CpuStorage::I64(ref data) => {
                    if dims.is_empty() {
                        Ok(Literal::scalar(data[0]))
                    } else if dims.len() == 1 {
                        Ok(Literal::vec1(data))
                    } else {
                        Ok(
                            Literal::create_from_shape_and_untyped_data(ElementType::S64, &dims, unsafe {
                                std::slice::from_raw_parts(
                                    data.as_ptr() as *const u8,
                                    data.len() * std::mem::size_of::<i64>(),
                                )
                            })
                            .map_err(xla_error_to_hodu_error)?,
                        )
                    }
                },
                _ => Err(HoduError::InternalError(
                    "Unsupported tensor data type in XLA conversion".to_string(),
                )),
            }
        })
    }

    /// Convert XLA literal to Hodu tensor
    fn literal_to_tensor(&self, literal: &Literal, dtype: DType) -> HoduResult<Tensor> {
        use crate::backends::be_hodu::cpu::storage::CpuStorage;
        use crate::backends::be_hodu::storage::HoduStorage;

        let array_shape = literal
            .array_shape()
            .map_err(|e| HoduError::InternalError(format!("Failed to get array shape from literal: {:?}", e)))?;

        let dims: Vec<usize> = array_shape.dims().iter().map(|&d| d as usize).collect();

        let cpu_storage = match dtype {
            DType::BOOL => {
                // Convert from u8 back to bool
                let u8_data = literal.to_vec::<u8>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract u8 data for bool from literal: {:?}", e))
                })?;
                let bool_data: Vec<bool> = u8_data.iter().map(|&b| b != 0).collect();
                CpuStorage::BOOL(bool_data)
            },
            DType::BF16 => {
                let data = literal.to_vec::<half::bf16>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract bf16 data from literal: {:?}", e))
                })?;
                CpuStorage::BF16(data)
            },
            DType::F16 => {
                let data = literal.to_vec::<half::f16>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract f16 data from literal: {:?}", e))
                })?;
                CpuStorage::F16(data)
            },
            DType::F32 => {
                let data = literal.to_vec::<f32>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract f32 data from literal: {:?}", e))
                })?;
                CpuStorage::F32(data)
            },
            DType::F64 => {
                let data = literal.to_vec::<f64>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract f64 data from literal: {:?}", e))
                })?;
                CpuStorage::F64(data)
            },
            DType::U8 => {
                let data = literal.to_vec::<u8>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract u8 data from literal: {:?}", e))
                })?;
                CpuStorage::U8(data)
            },
            DType::U16 => {
                let data = literal.to_vec::<u16>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract u16 data from literal: {:?}", e))
                })?;
                CpuStorage::U16(data)
            },
            DType::U32 => {
                let data = literal.to_vec::<u32>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract u32 data from literal: {:?}", e))
                })?;
                CpuStorage::U32(data)
            },
            DType::U64 => {
                let data = literal.to_vec::<u64>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract u64 data from literal: {:?}", e))
                })?;
                CpuStorage::U64(data)
            },
            DType::I8 => {
                let data = literal.to_vec::<i8>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract i8 data from literal: {:?}", e))
                })?;
                CpuStorage::I8(data)
            },
            DType::I16 => {
                let data = literal.to_vec::<i16>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract i16 data from literal: {:?}", e))
                })?;
                CpuStorage::I16(data)
            },
            DType::I32 => {
                let data = literal.to_vec::<i32>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract i32 data from literal: {:?}", e))
                })?;
                CpuStorage::I32(data)
            },
            DType::I64 => {
                let data = literal.to_vec::<i64>().map_err(|e| {
                    HoduError::InternalError(format!("Failed to extract i64 data from literal: {:?}", e))
                })?;
                CpuStorage::I64(data)
            },
            _ => {
                return Err(HoduError::InternalError(format!(
                    "Unsupported dtype for XLA conversion: {:?}",
                    dtype
                )));
            },
        };

        let hodu_storage = HoduStorage::CPU(cpu_storage);
        let layout = Layout::from_shape(&dims);
        let tensor = from_storage(hodu_storage, layout, true);
        Ok(tensor)
    }

    pub fn new(device: Device) -> HoduResult<Self> {
        let client = match device {
            Device::CPU => {
                Arc::new(ThreadSafeClient(PjRtClient::cpu().map_err(|e| {
                    HoduError::InternalError(format!("Failed to create CPU client: {:?}", e))
                })?))
            },
            Device::CUDA(_idx) => {
                #[cfg(feature = "cuda")]
                {
                    Arc::new(ThreadSafeClient(PjRtClient::gpu(_idx as f64, true).map_err(|e| {
                        HoduError::InternalError(format!("Failed to create GPU client for device {}: {:?}", _idx, e))
                    })?))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(HoduError::InternalError("CUDA support not compiled".to_string()));
                }
            },
            _ => Arc::new(ThreadSafeClient(PjRtClient::cpu().map_err(|e| {
                HoduError::InternalError(format!("Failed to create fallback CPU client: {:?}", e))
            })?)),
        };

        Ok(Self { client, device })
    }
}

unsafe impl Send for XlaExecutor {}
unsafe impl Sync for XlaExecutor {}

impl ExecutorT for XlaExecutor {
    type CompiledScript = XlaCompiledScript;

    fn backend_name(&self) -> &'static str {
        "xla"
    }

    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::CPU]
    }

    fn current_device(&self) -> Device {
        self.device
    }

    fn compile(&mut self, script: &Script, _options: CompileOptions) -> HoduResult<Self::CompiledScript> {
        let script_ir = script
            .get_ir()
            .ok_or_else(|| HoduError::ScriptValidationFailed("Script has no IR".to_string()))?;
        script_ir.validate().map_err(HoduError::ScriptValidationFailed)?;

        // Create input and output mappings exactly like HoduExecutor
        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();

        for input in &script_ir.graph.metadata.inputs {
            input_mapping.insert(input.name.clone(), input.tensor_id);
        }
        for output in &script_ir.graph.metadata.outputs {
            output_mapping.insert(output.name.clone(), output.tensor_id);
        }

        let tensor_layouts = self.collect_tensor_layouts(script_ir);
        let tensor_dtypes = self.collect_tensor_dtypes(script_ir);

        // Create dummy executable first
        let dummy_client = ThreadSafeClient(PjRtClient::cpu().map_err(xla_error_to_hodu_error)?);
        let dummy_builder = XlaBuilder::new("dummy");
        let dummy_param = dummy_builder
            .parameter(0, ElementType::F32, &[1], "dummy")
            .map_err(xla_error_to_hodu_error)?;
        let dummy_computation = dummy_param.build().map_err(xla_error_to_hodu_error)?;
        let dummy_executable = dummy_client
            .compile(&dummy_computation)
            .map_err(xla_error_to_hodu_error)?;

        let compiled_script = XlaCompiledScript {
            executable: Arc::new(ThreadSafeExecutable(dummy_executable)), // Dummy executable, will be replaced
            input_mapping,
            output_mapping,
            tensor_layouts,
            tensor_dtypes,
        };

        // Build the actual XLA computation
        let executable = self.build_xla_computation(&compiled_script, script_ir)?;

        Ok(XlaCompiledScript {
            executable,
            input_mapping: compiled_script.input_mapping,
            output_mapping: compiled_script.output_mapping,
            tensor_layouts: compiled_script.tensor_layouts,
            tensor_dtypes: compiled_script.tensor_dtypes,
        })
    }

    fn execute(&self, compiled: &Self::CompiledScript, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
        // Convert inputs to XLA literals using input_mapping (exactly like HoduExecutor)
        let mut xla_inputs = Vec::new();

        // Get input names in consistent order (sort for consistent ordering)
        let mut input_names: Vec<_> = compiled.input_mapping.keys().cloned().collect();
        input_names.sort();

        for input_name in input_names.iter() {
            let tensor = inputs
                .get(input_name.as_str())
                .ok_or_else(|| HoduError::InternalError(format!("Input {} not found", input_name)))?;

            let literal = self.tensor_to_literal(tensor).map_err(|e| {
                HoduError::InternalError(format!("Failed to convert input {} to literal: {}", input_name, e))
            })?;
            xla_inputs.push(literal);
        }

        // Execute the computation
        let result_buffers = compiled
            .executable
            .execute::<xla::Literal>(&xla_inputs)
            .map_err(|e| HoduError::InternalError(format!("Failed to execute XLA computation: {:?}", e)))?;

        // Convert results back to tensors using output_mapping
        let mut outputs = HashMap::new();

        // Get output names in consistent order
        let output_names: Vec<_> = compiled.output_mapping.keys().cloned().collect();

        for (i, output_name) in output_names.iter().enumerate() {
            let result_literal = result_buffers[0][i]
                .to_literal_sync()
                .map_err(|e| HoduError::InternalError(format!("Failed to convert result {} to literal: {:?}", i, e)))?;

            // Get expected dtype from tensor_dtypes mapping
            let output_tensor_id = compiled
                .output_mapping
                .get(output_name)
                .ok_or_else(|| HoduError::InternalError(format!("Output tensor ID not found for {}", output_name)))?;
            let expected_dtype = compiled
                .tensor_dtypes
                .get(output_tensor_id)
                .copied()
                .unwrap_or(DType::F32); // Default to F32 if not found

            let tensor = self
                .literal_to_tensor(&result_literal, expected_dtype)
                .map_err(|e| HoduError::InternalError(format!("Failed to convert literal {} to tensor: {}", i, e)))?;

            outputs.insert(output_name.clone(), tensor);
        }

        Ok(outputs)
    }

    fn cleanup(&mut self) -> HoduResult<()> {
        Ok(())
    }
}
