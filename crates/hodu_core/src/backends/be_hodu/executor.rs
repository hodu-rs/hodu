use crate::{
    backends::{
        be_hodu::{
            metal::storage::MetalStorage,
            storage::{HoduStorage, HoduStorageT},
        },
        executor::{CompileOptions, ExecutionInputs, ExecutionOutputs, ExecutorT},
        op::{
            BinaryLogicalOp, BinaryOp, CastOp, CmpOp, CmpScalarOp, ConvOp, IndexingOp, MatrixOp, MemoryOp, Op, ShapeOp,
            UnaryLogicalOp, UnaryOp, UnaryScalarOp, WindowingOp,
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

type SharedStorage = Arc<HoduStorage>;

#[derive(Debug)]
pub struct HoduExecutor {
    current_device: Device,
}

#[derive(Debug)]
pub struct HoduCompiledScript {
    execution_plan: Vec<CompiledNode>,
    input_mapping: HashMap<String, TensorId>,
    output_mapping: HashMap<String, TensorId>,
    constant_storage: HashMap<TensorId, SharedStorage>,
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
        let mut compiled_nodes = Vec::with_capacity(script_ir.graph.topology.nodes.len());

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

        // Build execution order lookup map for O(1) access
        #[cfg(feature = "std")]
        let execution_order_map: HashMap<NodeId, usize> = script_ir
            .graph
            .execution_plan
            .execution_order
            .iter()
            .enumerate()
            .map(|(index, &node_id)| (node_id, index))
            .collect();

        #[cfg(not(feature = "std"))]
        let execution_order_map: HashMap<NodeId, usize> = script_ir
            .graph
            .execution_plan
            .execution_order
            .iter()
            .enumerate()
            .map(|(index, &node_id)| (node_id, index))
            .collect();

        compiled_nodes.sort_by_key(|node| execution_order_map.get(&node.id).copied().unwrap_or(usize::MAX));

        Ok(compiled_nodes)
    }

    fn collect_tensor_layouts(&self, script_ir: &ScriptIR) -> HashMap<TensorId, Layout> {
        #[cfg(feature = "std")]
        let mut tensor_layouts = {
            // Estimate capacity: node layouts + inputs + outputs
            let estimated_layout_count = script_ir.graph.topology.nodes.len() * 2
                + script_ir.graph.metadata.inputs.len()
                + script_ir.graph.metadata.outputs.len();
            HashMap::with_capacity(estimated_layout_count)
        };
        #[cfg(not(feature = "std"))]
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
                    // Avoid Vec allocation by collecting into a small stack array when possible
                    let shape_usize: Vec<usize> = shape.iter().map(|s| s.unwrap_or(1)).collect();
                    tensor_layouts.insert(input.tensor_id, Layout::from_shape(&shape_usize));
                }
            }
        }

        for output in &script_ir.graph.metadata.outputs {
            #[cfg(not(feature = "std"))]
            pub use alloc::collections::btree_map::Entry;
            #[cfg(feature = "std")]
            pub use std::collections::hash_map::Entry;
            // Don't overwrite output layouts if they already exist from node processing
            // This preserves transpose and other view operation layouts
            if let Entry::Vacant(e) = tensor_layouts.entry(output.tensor_id) {
                if let Some(tensor_info) = script_ir.graph.metadata.tensor_info.get(&output.tensor_id) {
                    if let Some(ref shape) = tensor_info.shape {
                        // Avoid Vec allocation by collecting into a small stack array when possible
                        let shape_usize: Vec<usize> = shape.iter().map(|s| s.unwrap_or(1)).collect();
                        e.insert(Layout::from_shape(&shape_usize));
                    }
                }
            }
        }

        tensor_layouts
    }

    fn collect_tensor_dtypes(&self, script_ir: &ScriptIR, script: &Script) -> HashMap<TensorId, DType> {
        let mut tensor_dtypes = HashMap::new();

        for (&tensor_id, tensor_info) in &script_ir.graph.metadata.tensor_info {
            if let Some(dtype) = tensor_info.dtype {
                tensor_dtypes.insert(tensor_id, dtype);
            }
        }

        // Override dtypes for input tensors with actual runtime input dtypes
        let runtime_inputs = script.get_inputs();
        for input in &script_ir.graph.metadata.inputs {
            if let Some(tensor) = runtime_inputs.get(&input.name) {
                let actual_dtype = tensor.get_dtype();
                tensor_dtypes.insert(input.tensor_id, actual_dtype);
            }
        }

        tensor_dtypes
    }

    fn prepare_constant_storage(&self, script_ir: &ScriptIR) -> HoduResult<HashMap<TensorId, SharedStorage>> {
        #[cfg(feature = "std")]
        let mut constant_storage = HashMap::with_capacity(script_ir.graph.metadata.constants.len());
        #[cfg(not(feature = "std"))]
        let mut constant_storage = HashMap::new();

        for (tensor_id, constant_node) in &script_ir.graph.metadata.constants {
            match self.current_device {
                Device::CPU => {
                    let storage = self.convert_constant_to_cpu_storage(constant_node)?;
                    constant_storage.insert(*tensor_id, Arc::new(HoduStorage::CPU(storage)));
                },
                Device::CUDA(_) => {
                    // TODO: Convert to CUDA storage
                    return Err(HoduError::InternalError(
                        "CUDA constant conversion not implemented".to_string(),
                    ));
                },
                Device::Metal => {
                    let cpu_storage = self.convert_constant_to_cpu_storage(constant_node)?;
                    let metal_storage = MetalStorage::from_cpu_storage(&cpu_storage)?;
                    constant_storage.insert(*tensor_id, Arc::new(HoduStorage::Metal(metal_storage)));
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
            #[cfg(all(feature = "serde", feature = "std"))]
            Some(CompressionType::Gzip) => {
                use std::io::Read;
                let mut decoder = flate2::read::GzDecoder::new(&constant.data[..]);
                let mut decompressed = Vec::new();
                decoder
                    .read_to_end(&mut decompressed)
                    .map_err(|e| HoduError::DecompressionError(e.to_string()))?;
                decompressed
            },
            #[cfg(not(all(feature = "serde", feature = "std")))]
            Some(CompressionType::Gzip) => {
                return Err(HoduError::InternalError(
                    "Gzip decompression requires both 'serde' and 'std' features to be enabled".to_string(),
                ));
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
            #[cfg(feature = "metal")]
            (Device::Metal, storage) => match storage {
                HoduStorage::Metal(metal_storage) => Ok(HoduStorage::Metal(metal_storage.clone())),
                HoduStorage::CPU(cpu_storage) => {
                    use crate::backends::be_hodu::metal::storage::MetalStorage;
                    MetalStorage::from_cpu_storage(cpu_storage).map(HoduStorage::Metal)
                },
            },
            #[cfg(not(feature = "metal"))]
            (Device::Metal, _) => Err(HoduError::InternalError("Metal feature not enabled".to_string())),
        })
    }

    fn execute_node(
        &self,
        compiled_node: &CompiledNode,
        tensor_storage: &HashMap<TensorId, SharedStorage>,
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
            Op::Reduce(reduce_op, tensor_id, keep_dim, dims_scalars) => {
                let input_storage = tensor_storage
                    .get(tensor_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {tensor_id:?} not found")))?;

                let input_layout = compiled_node
                    .input_layouts
                    .first()
                    .ok_or_else(|| HoduError::InternalError("Reduce operation requires input layout".to_string()))?;

                // Extract dimensions from scalar array
                let dims: Vec<usize> = dims_scalars.iter().map(|scalar| scalar.to_u32() as usize).collect();

                input_storage.reduce(*reduce_op, input_layout, &dims, *keep_dim)
            },
            Op::Concat(_concat_op, input_tensor_ids, params) => {
                // Extract dimension from params
                let dim = params
                    .first()
                    .ok_or_else(|| {
                        HoduError::InternalError("Concat operation requires dimension parameter".to_string())
                    })?
                    .to_u32() as usize;

                // Get all input storages
                let input_storages: Vec<&SharedStorage> = input_tensor_ids
                    .iter()
                    .map(|tensor_id| {
                        tensor_storage
                            .get(tensor_id)
                            .ok_or_else(|| HoduError::InternalError(format!("Input tensor {tensor_id:?} not found")))
                    })
                    .collect::<HoduResult<Vec<_>>>()?;

                // Get all input layouts
                let layouts: Vec<&Layout> = compiled_node.input_layouts.iter().collect();

                // Call concat on first storage with remaining storages
                if input_storages.is_empty() {
                    return Err(HoduError::InternalError(
                        "Concat requires at least one input".to_string(),
                    ));
                }

                // Dereference Arc wrappers to get &HoduStorage
                let storage_refs: Vec<&HoduStorage> = input_storages.iter().map(|s| s.as_ref()).collect();
                storage_refs[0].concat(&storage_refs[1..], &layouts, dim)
            },
            Op::Split(_split_op, input_tensor_id, size_scalars, _output_index) => {
                let input_storage = tensor_storage
                    .get(input_tensor_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {input_tensor_id:?} not found")))?;

                let input_layout = compiled_node
                    .input_layouts
                    .first()
                    .ok_or_else(|| HoduError::InternalError("Split operation requires input layout".to_string()))?;

                // Extract dimension by comparing input and output shapes
                let output_layout = compiled_node
                    .output_layouts
                    .first()
                    .ok_or_else(|| HoduError::InternalError("Split operation requires output layout".to_string()))?;

                let input_shape = input_layout.get_shape();
                let output_shape = output_layout.get_shape();

                let mut dim = None;
                for (i, (&in_size, &out_size)) in input_shape.iter().zip(output_shape.iter()).enumerate() {
                    if in_size != out_size {
                        dim = Some(i);
                        break;
                    }
                }

                let dim =
                    dim.ok_or_else(|| HoduError::InternalError("Could not determine split dimension".to_string()))?;

                // Extract sizes from size_scalars (skip first element which is the dimension)
                let sizes: Vec<usize> = size_scalars.iter().skip(1).map(|s| s.to_u32() as usize).collect();

                // Call split on storage and return the specific output based on output_index
                // Note: The split returns all splits, but we store all of them with different output indices
                // Actually, for Split operation, each node produces one output, not all splits
                // So we need to return all splits and let the execution plan handle storing them
                input_storage.split(input_layout, dim, &sizes)?;

                // Return the first split storage (the actual split results are handled by the node execution)
                // This is a bit tricky because split returns Vec<Storage> but we need to return single Storage
                // Let's check how the split operation is supposed to work...
                // Looking at the Op::Split signature: Op::Split(SplitOp, TensorId, Vec<Scalar>, usize)
                // The usize is the output_index, so each Split node produces one output

                let splits = input_storage.split(input_layout, dim, &sizes)?;
                let output_index = match &compiled_node.operation {
                    Op::Split(_, _, _, idx) => *idx,
                    _ => 0,
                };

                if output_index >= splits.len() {
                    return Err(HoduError::InternalError(format!(
                        "Split output index {} out of bounds (total splits: {})",
                        output_index,
                        splits.len()
                    )));
                }

                // Return only the specific split for this output
                Ok(splits[output_index].clone())
            },
            Op::Indexing(indexing_op, tensor_ids, params) => {
                match indexing_op {
                    IndexingOp::IndexSelect | IndexingOp::Gather => {
                        if tensor_ids.len() != 2 {
                            return Err(HoduError::InternalError(format!(
                                "{:?} requires 2 tensors, got {}",
                                indexing_op,
                                tensor_ids.len()
                            )));
                        }

                        let self_storage = tensor_storage
                            .get(&tensor_ids[0])
                            .ok_or_else(|| HoduError::InternalError(format!("Tensor {:?} not found", tensor_ids[0])))?;
                        let indices_storage = tensor_storage
                            .get(&tensor_ids[1])
                            .ok_or_else(|| HoduError::InternalError(format!("Tensor {:?} not found", tensor_ids[1])))?;

                        let self_layout = &compiled_node.input_layouts[0];
                        let indices_layout = &compiled_node.input_layouts[1];

                        // Extract dim from params
                        let dim = params
                            .first()
                            .ok_or_else(|| {
                                HoduError::InternalError(format!("{:?} requires dimension parameter", indexing_op))
                            })?
                            .to_u32() as usize;

                        match indexing_op {
                            IndexingOp::IndexSelect => {
                                self_storage.index_select(self_layout, indices_storage, indices_layout, dim)
                            },
                            IndexingOp::Gather => {
                                self_storage.gather(self_layout, indices_storage, indices_layout, dim)
                            },
                            _ => unreachable!(),
                        }
                    },
                    IndexingOp::IndexPut
                    | IndexingOp::Scatter
                    | IndexingOp::ScatterAdd
                    | IndexingOp::ScatterMax
                    | IndexingOp::ScatterMin => {
                        if tensor_ids.len() != 3 {
                            return Err(HoduError::InternalError(format!(
                                "{:?} requires 3 tensors, got {}",
                                indexing_op,
                                tensor_ids.len()
                            )));
                        }

                        let self_storage = tensor_storage
                            .get(&tensor_ids[0])
                            .ok_or_else(|| HoduError::InternalError(format!("Tensor {:?} not found", tensor_ids[0])))?;
                        let indices_storage = tensor_storage
                            .get(&tensor_ids[1])
                            .ok_or_else(|| HoduError::InternalError(format!("Tensor {:?} not found", tensor_ids[1])))?;
                        let src_storage = tensor_storage
                            .get(&tensor_ids[2])
                            .ok_or_else(|| HoduError::InternalError(format!("Tensor {:?} not found", tensor_ids[2])))?;

                        let self_layout = &compiled_node.input_layouts[0];
                        let indices_layout = &compiled_node.input_layouts[1];
                        let src_layout = &compiled_node.input_layouts[2];

                        // Extract dim from params
                        let dim = params
                            .first()
                            .ok_or_else(|| {
                                HoduError::InternalError(format!("{:?} requires dimension parameter", indexing_op))
                            })?
                            .to_u32() as usize;

                        match indexing_op {
                            IndexingOp::IndexPut => self_storage.index_put(
                                self_layout,
                                indices_storage,
                                indices_layout,
                                src_storage,
                                src_layout,
                                dim,
                            ),
                            IndexingOp::Scatter => self_storage.scatter(
                                self_layout,
                                indices_storage,
                                indices_layout,
                                src_storage,
                                src_layout,
                                dim,
                            ),
                            IndexingOp::ScatterAdd => self_storage.scatter_add(
                                self_layout,
                                indices_storage,
                                indices_layout,
                                src_storage,
                                src_layout,
                                dim,
                            ),
                            IndexingOp::ScatterMax => self_storage.scatter_max(
                                self_layout,
                                indices_storage,
                                indices_layout,
                                src_storage,
                                src_layout,
                                dim,
                            ),
                            IndexingOp::ScatterMin => self_storage.scatter_min(
                                self_layout,
                                indices_storage,
                                indices_layout,
                                src_storage,
                                src_layout,
                                dim,
                            ),
                            _ => unreachable!(),
                        }
                    },
                }
            },
            Op::Conv(conv_op, input_id, weight_id, params) => {
                let input_storage = tensor_storage
                    .get(input_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {input_id:?} not found")))?;
                let weight_storage = tensor_storage
                    .get(weight_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Weight tensor {weight_id:?} not found")))?;
                let input_layout = &compiled_node.input_layouts[0];
                let weight_layout = &compiled_node.input_layouts[1];

                self.execute_conv_op(
                    *conv_op,
                    input_storage,
                    weight_storage,
                    input_layout,
                    weight_layout,
                    params,
                )
            },
            Op::Windowing(windowing_op, tensor_id, params) => {
                let input_storage = tensor_storage
                    .get(tensor_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {tensor_id:?} not found")))?;
                let input_layout = &compiled_node.input_layouts[0];

                self.execute_windowing_op(*windowing_op, input_storage, input_layout, params)
            },
            Op::Shape(shape_op, tensor_id) => {
                let input_storage = tensor_storage
                    .get(tensor_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {tensor_id:?} not found")))?;

                self.execute_shape_op(*shape_op, input_storage)
            },
            Op::Cast(cast_op, tensor_id) => {
                let input_storage = tensor_storage
                    .get(tensor_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {tensor_id:?} not found")))?;

                self.execute_cast_op(
                    *cast_op,
                    input_storage,
                    &compiled_node.input_layouts[0],
                    compiled_node,
                    compiled,
                )
            },
            Op::Memory(memory_op, tensor_id) => {
                let input_storage = tensor_storage
                    .get(tensor_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {tensor_id:?} not found")))?;

                self.execute_memory_op(*memory_op, input_storage, compiled_node)
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
        lhs_storage: &SharedStorage,
        rhs_storage: &SharedStorage,
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
        lhs_storage: &SharedStorage,
        rhs_storage: &SharedStorage,
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
        lhs_storage: &SharedStorage,
        rhs_storage: &SharedStorage,
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
        tensor_storage: &SharedStorage,
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
        input_storage: &SharedStorage,
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
        input_storage: &SharedStorage,
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
        input_storage: &SharedStorage,
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
        lhs_storage: &SharedStorage,
        rhs_storage: &SharedStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> HoduResult<HoduStorage> {
        match matrix_op {
            MatrixOp::Matmul => lhs_storage.matmul(rhs_storage, lhs_layout, rhs_layout),
            MatrixOp::Dot => lhs_storage.dot(rhs_storage, lhs_layout, rhs_layout),
        }
    }

    fn execute_conv_op(
        &self,
        conv_op: ConvOp,
        input_storage: &SharedStorage,
        weight_storage: &SharedStorage,
        input_layout: &Layout,
        weight_layout: &Layout,
        params: &[Scalar],
    ) -> HoduResult<HoduStorage> {
        use crate::backends::op::conv::*;

        match conv_op {
            ConvOp::Conv1d => {
                if params.len() < 8 {
                    return Err(HoduError::InternalError("Conv1d requires 8 parameters".to_string()));
                }
                let conv_params = ParamsConv1D {
                    batch_size: params[0].to_u32() as usize,
                    length_input: params[1].to_u32() as usize,
                    channels_output: params[2].to_u32() as usize,
                    channels_input: params[3].to_u32() as usize,
                    kernel_size: params[4].to_u32() as usize,
                    padding: params[5].to_u32() as usize,
                    stride: params[6].to_u32() as usize,
                    dilation: params[7].to_u32() as usize,
                };
                input_storage.conv1d(weight_storage, input_layout, weight_layout, &conv_params)
            },
            ConvOp::Conv2d => {
                if params.len() < 10 {
                    return Err(HoduError::InternalError("Conv2d requires 10 parameters".to_string()));
                }
                let conv_params = ParamsConv2D {
                    batch_size: params[0].to_u32() as usize,
                    input_height: params[1].to_u32() as usize,
                    input_width: params[2].to_u32() as usize,
                    kernel_height: params[3].to_u32() as usize,
                    kernel_width: params[4].to_u32() as usize,
                    channels_output: params[5].to_u32() as usize,
                    channels_input: params[6].to_u32() as usize,
                    padding: params[7].to_u32() as usize,
                    stride: params[8].to_u32() as usize,
                    dilation: params[9].to_u32() as usize,
                };
                input_storage.conv2d(weight_storage, input_layout, weight_layout, &conv_params)
            },
            ConvOp::Conv3d => {
                if params.len() < 12 {
                    return Err(HoduError::InternalError("Conv3d requires 12 parameters".to_string()));
                }
                let conv_params = ParamsConv3D {
                    batch_size: params[0].to_u32() as usize,
                    input_depth: params[1].to_u32() as usize,
                    input_height: params[2].to_u32() as usize,
                    input_width: params[3].to_u32() as usize,
                    kernel_depth: params[4].to_u32() as usize,
                    kernel_height: params[5].to_u32() as usize,
                    kernel_width: params[6].to_u32() as usize,
                    channels_output: params[7].to_u32() as usize,
                    channels_input: params[8].to_u32() as usize,
                    padding: params[9].to_u32() as usize,
                    stride: params[10].to_u32() as usize,
                    dilation: params[11].to_u32() as usize,
                };
                input_storage.conv3d(weight_storage, input_layout, weight_layout, &conv_params)
            },
            ConvOp::ConvTranspose1d => {
                if params.len() < 9 {
                    return Err(HoduError::InternalError(
                        "ConvTranspose1d requires 9 parameters".to_string(),
                    ));
                }
                let conv_params = ParamsConvTranspose1D {
                    batch_size: params[0].to_u32() as usize,
                    length_input: params[1].to_u32() as usize,
                    channels_output: params[2].to_u32() as usize,
                    channels_input: params[3].to_u32() as usize,
                    kernel_size: params[4].to_u32() as usize,
                    padding: params[5].to_u32() as usize,
                    output_padding: params[6].to_u32() as usize,
                    stride: params[7].to_u32() as usize,
                    dilation: params[8].to_u32() as usize,
                };
                input_storage.conv_transpose1d(weight_storage, input_layout, weight_layout, &conv_params)
            },
            ConvOp::ConvTranspose2d => {
                if params.len() < 11 {
                    return Err(HoduError::InternalError(
                        "ConvTranspose2d requires 11 parameters".to_string(),
                    ));
                }
                let conv_params = ParamsConvTranspose2D {
                    batch_size: params[0].to_u32() as usize,
                    input_height: params[1].to_u32() as usize,
                    input_width: params[2].to_u32() as usize,
                    kernel_height: params[3].to_u32() as usize,
                    kernel_width: params[4].to_u32() as usize,
                    channels_output: params[5].to_u32() as usize,
                    channels_input: params[6].to_u32() as usize,
                    padding: params[7].to_u32() as usize,
                    output_padding: params[8].to_u32() as usize,
                    stride: params[9].to_u32() as usize,
                    dilation: params[10].to_u32() as usize,
                };
                input_storage.conv_transpose2d(weight_storage, input_layout, weight_layout, &conv_params)
            },
            ConvOp::ConvTranspose3d => {
                if params.len() < 13 {
                    return Err(HoduError::InternalError(
                        "ConvTranspose3d requires 13 parameters".to_string(),
                    ));
                }
                let conv_params = ParamsConvTranspose3D {
                    batch_size: params[0].to_u32() as usize,
                    input_depth: params[1].to_u32() as usize,
                    input_height: params[2].to_u32() as usize,
                    input_width: params[3].to_u32() as usize,
                    kernel_depth: params[4].to_u32() as usize,
                    kernel_height: params[5].to_u32() as usize,
                    kernel_width: params[6].to_u32() as usize,
                    channels_output: params[7].to_u32() as usize,
                    channels_input: params[8].to_u32() as usize,
                    padding: params[9].to_u32() as usize,
                    output_padding: params[10].to_u32() as usize,
                    stride: params[11].to_u32() as usize,
                    dilation: params[12].to_u32() as usize,
                };
                input_storage.conv_transpose3d(weight_storage, input_layout, weight_layout, &conv_params)
            },
            ConvOp::Conv1dGradWeight => {
                if params.len() < 8 {
                    return Err(HoduError::InternalError(
                        "Conv1dGradWeight requires 8 parameters".to_string(),
                    ));
                }
                let conv_params = ParamsConv1D {
                    batch_size: params[0].to_u32() as usize,
                    length_input: params[1].to_u32() as usize,
                    channels_output: params[2].to_u32() as usize,
                    channels_input: params[3].to_u32() as usize,
                    kernel_size: params[4].to_u32() as usize,
                    padding: params[5].to_u32() as usize,
                    stride: params[6].to_u32() as usize,
                    dilation: params[7].to_u32() as usize,
                };
                input_storage.conv1d_grad_weight(weight_storage, input_layout, weight_layout, &conv_params)
            },
            ConvOp::Conv2dGradWeight => {
                if params.len() < 10 {
                    return Err(HoduError::InternalError(
                        "Conv2dGradWeight requires 10 parameters".to_string(),
                    ));
                }
                let conv_params = ParamsConv2D {
                    batch_size: params[0].to_u32() as usize,
                    input_height: params[1].to_u32() as usize,
                    input_width: params[2].to_u32() as usize,
                    kernel_height: params[3].to_u32() as usize,
                    kernel_width: params[4].to_u32() as usize,
                    channels_output: params[5].to_u32() as usize,
                    channels_input: params[6].to_u32() as usize,
                    padding: params[7].to_u32() as usize,
                    stride: params[8].to_u32() as usize,
                    dilation: params[9].to_u32() as usize,
                };
                input_storage.conv2d_grad_weight(weight_storage, input_layout, weight_layout, &conv_params)
            },
            ConvOp::Conv3dGradWeight => {
                if params.len() < 12 {
                    return Err(HoduError::InternalError(
                        "Conv3dGradWeight requires 12 parameters".to_string(),
                    ));
                }
                let conv_params = ParamsConv3D {
                    batch_size: params[0].to_u32() as usize,
                    input_depth: params[1].to_u32() as usize,
                    input_height: params[2].to_u32() as usize,
                    input_width: params[3].to_u32() as usize,
                    kernel_depth: params[4].to_u32() as usize,
                    kernel_height: params[5].to_u32() as usize,
                    kernel_width: params[6].to_u32() as usize,
                    channels_output: params[7].to_u32() as usize,
                    channels_input: params[8].to_u32() as usize,
                    padding: params[9].to_u32() as usize,
                    stride: params[10].to_u32() as usize,
                    dilation: params[11].to_u32() as usize,
                };
                input_storage.conv3d_grad_weight(weight_storage, input_layout, weight_layout, &conv_params)
            },
            ConvOp::ConvTranspose1dGradWeight => {
                if params.len() < 9 {
                    return Err(HoduError::InternalError(
                        "ConvTranspose1dGradWeight requires 9 parameters".to_string(),
                    ));
                }
                let conv_params = ParamsConvTranspose1D {
                    batch_size: params[0].to_u32() as usize,
                    length_input: params[1].to_u32() as usize,
                    channels_output: params[2].to_u32() as usize,
                    channels_input: params[3].to_u32() as usize,
                    kernel_size: params[4].to_u32() as usize,
                    padding: params[5].to_u32() as usize,
                    stride: params[6].to_u32() as usize,
                    output_padding: params[7].to_u32() as usize,
                    dilation: params[8].to_u32() as usize,
                };
                input_storage.conv_transpose1d_grad_weight(weight_storage, input_layout, weight_layout, &conv_params)
            },
            ConvOp::ConvTranspose2dGradWeight => {
                if params.len() < 11 {
                    return Err(HoduError::InternalError(
                        "ConvTranspose2dGradWeight requires 11 parameters".to_string(),
                    ));
                }
                let conv_params = ParamsConvTranspose2D {
                    batch_size: params[0].to_u32() as usize,
                    input_height: params[1].to_u32() as usize,
                    input_width: params[2].to_u32() as usize,
                    kernel_height: params[3].to_u32() as usize,
                    kernel_width: params[4].to_u32() as usize,
                    channels_output: params[5].to_u32() as usize,
                    channels_input: params[6].to_u32() as usize,
                    padding: params[7].to_u32() as usize,
                    stride: params[8].to_u32() as usize,
                    output_padding: params[9].to_u32() as usize,
                    dilation: params[10].to_u32() as usize,
                };
                input_storage.conv_transpose2d_grad_weight(weight_storage, input_layout, weight_layout, &conv_params)
            },
            ConvOp::ConvTranspose3dGradWeight => {
                if params.len() < 13 {
                    return Err(HoduError::InternalError(
                        "ConvTranspose3dGradWeight requires 13 parameters".to_string(),
                    ));
                }
                let conv_params = ParamsConvTranspose3D {
                    batch_size: params[0].to_u32() as usize,
                    input_depth: params[1].to_u32() as usize,
                    input_height: params[2].to_u32() as usize,
                    input_width: params[3].to_u32() as usize,
                    kernel_depth: params[4].to_u32() as usize,
                    kernel_height: params[5].to_u32() as usize,
                    kernel_width: params[6].to_u32() as usize,
                    channels_output: params[7].to_u32() as usize,
                    channels_input: params[8].to_u32() as usize,
                    padding: params[9].to_u32() as usize,
                    output_padding: params[10].to_u32() as usize,
                    stride: params[11].to_u32() as usize,
                    dilation: params[12].to_u32() as usize,
                };
                input_storage.conv_transpose3d_grad_weight(weight_storage, input_layout, weight_layout, &conv_params)
            },
        }
    }

    fn execute_windowing_op(
        &self,
        windowing_op: WindowingOp,
        input_storage: &SharedStorage,
        input_layout: &Layout,
        params: &[Scalar],
    ) -> HoduResult<HoduStorage> {
        use crate::backends::op::window_reduction::WindowReduction;

        match windowing_op {
            WindowingOp::ReduceWindow => {
                // Parse parameters: window_shape, strides, padding, reduction_type
                // Parameters are packed as: [rank, window_shape..., strides..., padding_lo..., padding_hi..., reduction_type]
                if params.is_empty() {
                    return Err(HoduError::InternalError("ReduceWindow requires parameters".to_string()));
                }

                let rank = params[0].to_u32() as usize;
                let expected_len = 1 + rank + rank + rank + rank + 1; // rank + window_shape + strides + padding_lo + padding_hi + reduction_type

                if params.len() != expected_len {
                    return Err(HoduError::InternalError(format!(
                        "ReduceWindow requires {} parameters, got {}",
                        expected_len,
                        params.len()
                    )));
                }

                let mut offset = 1;

                // Extract window_shape
                let window_shape: Vec<usize> = params[offset..offset + rank]
                    .iter()
                    .map(|s| s.to_u32() as usize)
                    .collect();
                offset += rank;

                // Extract strides
                let strides: Vec<usize> = params[offset..offset + rank]
                    .iter()
                    .map(|s| s.to_u32() as usize)
                    .collect();
                offset += rank;

                // Extract padding (interleaved: lo, hi, lo, hi, ...)
                let mut padding: Vec<(usize, usize)> = Vec::with_capacity(rank);
                for _ in 0..rank {
                    let pad_lo = params[offset].to_u32() as usize;
                    let pad_hi = params[offset + 1].to_u32() as usize;
                    padding.push((pad_lo, pad_hi));
                    offset += 2;
                }

                // Extract reduction type
                let reduction_type_val = params[offset].to_u32();
                let reduction = match reduction_type_val {
                    0 => WindowReduction::Max,
                    1 => WindowReduction::Mean,
                    2 => WindowReduction::Sum,
                    3 => WindowReduction::Min,
                    _ => {
                        return Err(HoduError::InternalError(format!(
                            "Invalid reduction type: {}",
                            reduction_type_val
                        )))
                    },
                };

                input_storage.reduce_window(input_layout, &window_shape, &strides, &padding, reduction)
            },
        }
    }

    fn execute_shape_op(&self, _shape_op: ShapeOp, input_storage: &SharedStorage) -> HoduResult<HoduStorage> {
        // All shape operations (Reshape, Flatten, Squeeze, Unsqueeze, Broadcast, Transpose)
        // simply return the same storage as the layout changes are handled at the tensor level
        match input_storage.as_ref() {
            HoduStorage::CPU(cpu_storage) => Ok(HoduStorage::CPU(cpu_storage.clone())),
            HoduStorage::Metal(metal_storage) => Ok(HoduStorage::Metal(metal_storage.clone())),
        }
    }

    fn execute_cast_op(
        &self,
        cast_op: CastOp,
        input_storage: &SharedStorage,
        input_layout: &Layout,
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

                match input_storage.as_ref() {
                    HoduStorage::CPU(cpu_storage) => {
                        let converted_storage = cpu_storage.to_dtype(target_dtype, input_layout)?;
                        Ok(HoduStorage::CPU(converted_storage))
                    },
                    HoduStorage::Metal(metal_storage) => {
                        let converted_storage = metal_storage.to_dtype(target_dtype, input_layout)?;
                        Ok(HoduStorage::Metal(converted_storage))
                    },
                }
            },
        }
    }

    fn execute_memory_op(
        &self,
        memory_op: MemoryOp,
        input_storage: &SharedStorage,
        compiled_node: &CompiledNode,
    ) -> HoduResult<HoduStorage> {
        match memory_op {
            MemoryOp::Contiguous => {
                // Get the input layout from compiled node
                let input_layout = compiled_node
                    .input_layouts
                    .first()
                    .ok_or_else(|| HoduError::InternalError("Contiguous operation has no input layout".to_string()))?;

                // Execute contiguous operation on storage
                input_storage.contiguous(input_layout)
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
        let tensor_dtypes = self.collect_tensor_dtypes(script_ir, script);

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

        // Pre-allocate HashMap with estimated capacity (std only)
        #[cfg(feature = "std")]
        let mut tensor_storage: HashMap<TensorId, SharedStorage> = {
            let estimated_capacity = compiled.constant_storage.len() + inputs.len() + compiled.execution_plan.len();
            HashMap::with_capacity(estimated_capacity)
        };
        #[cfg(not(feature = "std"))]
        let mut tensor_storage: HashMap<TensorId, SharedStorage> = HashMap::new();

        // Insert constant storage (already wrapped in Arc - no cloning needed)
        for (&tensor_id, shared_storage) in &compiled.constant_storage {
            tensor_storage.insert(tensor_id, shared_storage.clone());
        }

        // Convert input tensors to storage
        for (input_name, input_tensor) in &inputs {
            if let Some(&tensor_id) = compiled.input_mapping.get(*input_name) {
                let storage = self.tensor_to_storage(input_tensor)?;
                tensor_storage.insert(tensor_id, Arc::new(storage));
            }
        }

        // Execute computation graph
        for compiled_node in &compiled.execution_plan {
            let result_storage = self.execute_node(compiled_node, &tensor_storage, compiled)?;

            // Insert result for all output tensors of this node
            let shared_result = Arc::new(result_storage);
            for &output_tensor_id in &compiled_node.output_tensors {
                tensor_storage.insert(output_tensor_id, shared_result.clone());
            }
        }

        // Prepare final outputs with pre-allocated capacity (std only)
        #[cfg(feature = "std")]
        let mut outputs = HashMap::with_capacity(compiled.output_mapping.len());
        #[cfg(not(feature = "std"))]
        let mut outputs = HashMap::new();
        for (output_name, &tensor_id) in &compiled.output_mapping {
            if let Some(storage) = tensor_storage.get(&tensor_id) {
                if let Some(layout) = compiled.tensor_layouts.get(&tensor_id) {
                    // Clone the storage data only when creating the final output tensor
                    // Use Arc::try_unwrap to avoid cloning when possible
                    let output_storage = match Arc::try_unwrap(Arc::clone(storage)) {
                        Ok(storage) => storage,
                        Err(shared_storage) => match shared_storage.as_ref() {
                            HoduStorage::CPU(cpu_storage) => HoduStorage::CPU(cpu_storage.clone()),
                            HoduStorage::Metal(metal_storage) => HoduStorage::Metal(metal_storage.clone()),
                        },
                    };
                    let output_tensor = from_storage(output_storage, layout.clone(), false);
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
