use super::{CompiledNode, HoduCompiledScript, HoduExecutor};
use crate::{
    be_hodu::storage::{HoduStorage, HoduStorageT},
    compat::*,
    error::{HoduError, HoduResult},
    executor::ExecutionInputs,
    op::{
        BinaryLogicalOp, BinaryOp, CastOp, CmpOp, CmpScalarOp, ConvOp, IndexingOp, MatrixOp, MemoryOp, Op, ShapeOp,
        ShapeScalarsOp, UnaryLogicalOp, UnaryOp, UnaryScalarOp, WindowingOp,
    },
    scalar::Scalar,
    tensor::{Tensor, TensorId},
    types::{device::Device, dtype::DType, layout::Layout},
};

type SharedStorage = Arc<HoduStorage>;

/// Runtime execution methods
impl HoduExecutor {
    pub(super) fn validate_inputs(
        &self,
        compiled: &HoduCompiledScript,
        inputs: &ExecutionInputs<'_>,
    ) -> HoduResult<()> {
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

    pub(super) fn tensor_to_storage(&self, tensor: &Tensor) -> HoduResult<HoduStorage> {
        tensor.with_storage(|storage| match (&self.current_device, storage) {
            (Device::CPU, storage) => storage.to_cpu_storage().map(HoduStorage::CPU),
            (Device::CUDA(_), _) => Err(HoduError::InternalError(
                "CUDA tensor conversion not implemented".to_string(),
            )),
            #[cfg(feature = "metal")]
            (Device::Metal, storage) => match storage {
                HoduStorage::Metal(metal_storage) => Ok(HoduStorage::Metal(metal_storage.clone())),
                HoduStorage::CPU(cpu_storage) => {
                    use crate::be_hodu::metal::storage::MetalStorage;
                    MetalStorage::from_cpu_storage(cpu_storage).map(HoduStorage::Metal)
                },
            },
            #[cfg(not(feature = "metal"))]
            (Device::Metal, _) => Err(HoduError::InternalError("Metal feature not enabled".to_string())),
        })
    }

    pub(super) fn execute_node(
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
            Op::ShapeScalars(shape_scalars_op, tensor_id, scalars) => {
                let input_storage = tensor_storage
                    .get(tensor_id)
                    .ok_or_else(|| HoduError::InternalError(format!("Input tensor {tensor_id:?} not found")))?;

                self.execute_shape_scalars_op(*shape_scalars_op, input_storage, scalars)
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
        use crate::op::*;

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
        use crate::op::*;

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
        use crate::op::*;

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
        use crate::op::*;

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
        use crate::op::*;

        match unary_op {
            UnaryOp::Neg => input_storage.unary_impl::<Neg>(layout),
            UnaryOp::Abs => input_storage.unary_impl::<Abs>(layout),
            UnaryOp::Sign => input_storage.unary_impl::<Sign>(layout),
            UnaryOp::Square => input_storage.unary_impl::<Square>(layout),
            UnaryOp::Relu => input_storage.unary_impl::<Relu>(layout),
            UnaryOp::Sigmoid => input_storage.unary_impl::<Sigmoid>(layout),
            UnaryOp::Tanh => input_storage.unary_impl::<Tanh>(layout),
            UnaryOp::Gelu => input_storage.unary_impl::<Gelu>(layout),
            UnaryOp::Silu => input_storage.unary_impl::<Silu>(layout),
            UnaryOp::Mish => input_storage.unary_impl::<Mish>(layout),
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
        use crate::op::*;

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
        use crate::op::*;

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
            UnaryScalarOp::Prelu => input_storage.unary_scalar_impl::<Prelu>(layout, scalar),
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
        use crate::op::conv::*;

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
        use crate::op::window_reduction::WindowReduction;

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

    fn execute_shape_scalars_op(
        &self,
        _shape_scalars_op: ShapeScalarsOp,
        input_storage: &SharedStorage,
        _scalars: &[Scalar],
    ) -> HoduResult<HoduStorage> {
        // All shape scalars operations (Slice)
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
