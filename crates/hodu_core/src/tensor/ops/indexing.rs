use crate::{
    compat::*,
    error::HoduResult,
    ops::{IndexingOp, Op, OpParams},
    scalar::Scalar,
    script::builder,
    tensor::{create_builder_tensor, from_storage_with_context, gradient, register_operation_in_builder, Tensor},
    types::Layout,
    utils::valid::{
        validate_dtype_for_device, validate_dtype_for_op, validate_indices_dtype, validate_requires_grad_for_op,
        validate_same_device, validate_same_dtype,
    },
};

impl Tensor {
    pub fn index_select<D: Into<Scalar>>(&self, dim: D, indices: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices], Op::Indexing(IndexingOp::IndexSelect))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::IndexSelect))?;
        validate_indices_dtype(indices, Op::Indexing(IndexingOp::IndexSelect))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Indexing(IndexingOp::IndexSelect));

        let shape = self.shape();
        let shape_dims = shape.dims();
        let indices_size = indices.size();
        let mut output_dims = shape_dims.to_vec();
        output_dims[dim_usize] = indices_size;

        let result_layout = Layout::from_shape(&crate::types::Shape::from(output_dims));
        let self_layout = self.layout();
        let indices_layout = indices.layout();

        if builder::is_builder_active() {
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                Op::Indexing(IndexingOp::IndexSelect),
                Some(OpParams {
                    scalars: vec![dim_scalar],
                    ..Default::default()
                }),
                vec![self.id(), indices.id()],
                vec![result_id],
                vec![self_layout, indices_layout],
                vec![result_layout],
            )?;

            if requires_grad {
                gradient::record_operation_with_dims(
                    result_id,
                    Op::Indexing(IndexingOp::IndexSelect),
                    vec![self.id(), indices.id()],
                    vec![dim_scalar],
                    None,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    storage.call_ops_index_select(
                        &self_layout,
                        indices_storage,
                        &indices_layout,
                        dim_usize,
                        Op::Indexing(IndexingOp::IndexSelect),
                    )
                })
            })?;

            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Indexing(IndexingOp::IndexSelect);
                gradient::record_operation_with_dims(
                    result.id(),
                    op,
                    vec![self.id(), indices.id()],
                    vec![dim_scalar],
                    None,
                )?;
            }

            Ok(result)
        }
    }

    pub fn index_put<D: Into<Scalar>>(&self, dim: D, indices: &Self, values: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, values], Op::Indexing(IndexingOp::IndexPut))?;
        validate_same_dtype(&[self, values], Op::Indexing(IndexingOp::IndexPut))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::IndexPut))?;
        validate_indices_dtype(indices, Op::Indexing(IndexingOp::IndexPut))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Indexing(IndexingOp::IndexPut));

        let result_layout = self.layout();
        let indices_layout = indices.layout();
        let values_layout = values.layout();

        if builder::is_builder_active() {
            let requires_grad = (self.is_requires_grad() || values.is_requires_grad()) && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                Op::Indexing(IndexingOp::IndexPut),
                Some(OpParams {
                    scalars: vec![dim_scalar],
                    ..Default::default()
                }),
                vec![self.id(), indices.id(), values.id()],
                vec![result_id],
                vec![result_layout.clone(), indices_layout, values_layout],
                vec![result_layout.clone()],
            )?;

            if requires_grad {
                gradient::record_operation_with_dims(
                    result_id,
                    Op::Indexing(IndexingOp::IndexPut),
                    vec![self.id(), indices.id(), values.id()],
                    vec![dim_scalar],
                    None,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    values.with_storage(|values_storage| {
                        storage.call_ops_index_put(
                            &result_layout,
                            indices_storage,
                            &indices_layout,
                            values_storage,
                            &values_layout,
                            dim_usize,
                            Op::Indexing(IndexingOp::IndexPut),
                        )
                    })
                })
            })?;

            let requires_grad = (self.is_requires_grad() || values.is_requires_grad()) && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Indexing(IndexingOp::IndexPut);
                gradient::record_operation_with_dims(
                    result.id(),
                    op,
                    vec![self.id(), indices.id(), values.id()],
                    vec![dim_scalar],
                    None,
                )?;
            }

            Ok(result)
        }
    }

    pub fn gather<D: Into<Scalar>>(&self, dim: D, indices: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices], Op::Indexing(IndexingOp::Gather))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::Gather))?;
        validate_indices_dtype(indices, Op::Indexing(IndexingOp::Gather))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Indexing(IndexingOp::Gather));

        // Output has same shape as indices
        let self_layout = self.layout();
        let result_layout = indices.layout();

        if builder::is_builder_active() {
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                Op::Indexing(IndexingOp::Gather),
                Some(OpParams {
                    scalars: vec![dim_scalar],
                    ..Default::default()
                }),
                vec![self.id(), indices.id()],
                vec![result_id],
                vec![self_layout.clone(), result_layout.clone()],
                vec![result_layout],
            )?;

            if requires_grad {
                gradient::record_operation_with_dims(
                    result_id,
                    Op::Indexing(IndexingOp::Gather),
                    vec![self.id(), indices.id()],
                    vec![dim_scalar],
                    None,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    storage.call_ops_gather(
                        &self_layout,
                        indices_storage,
                        &result_layout,
                        dim_usize,
                        Op::Indexing(IndexingOp::Gather),
                    )
                })
            })?;

            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Indexing(IndexingOp::Gather);
                gradient::record_operation_with_dims(
                    result.id(),
                    op,
                    vec![self.id(), indices.id()],
                    vec![dim_scalar],
                    None,
                )?;
            }

            Ok(result)
        }
    }

    pub fn scatter<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], Op::Indexing(IndexingOp::Scatter))?;
        validate_same_dtype(&[self, src], Op::Indexing(IndexingOp::Scatter))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::Scatter))?;
        validate_indices_dtype(indices, Op::Indexing(IndexingOp::Scatter))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Indexing(IndexingOp::Scatter));

        // Output has same shape as self
        let result_layout = self.layout();
        let indices_layout = indices.layout();
        let src_layout = src.layout();

        if builder::is_builder_active() {
            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                Op::Indexing(IndexingOp::Scatter),
                Some(OpParams {
                    scalars: vec![dim_scalar],
                    ..Default::default()
                }),
                vec![self.id(), indices.id(), src.id()],
                vec![result_id],
                vec![result_layout.clone(), indices_layout.clone(), src_layout.clone()],
                vec![result_layout],
            )?;

            if requires_grad {
                gradient::record_operation_with_dims(
                    result_id,
                    Op::Indexing(IndexingOp::Scatter),
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                    None,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.call_ops_scatter(
                            &result_layout,
                            indices_storage,
                            &indices_layout,
                            src_storage,
                            &src_layout,
                            dim_usize,
                            Op::Indexing(IndexingOp::Scatter),
                        )
                    })
                })
            })?;

            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Indexing(IndexingOp::Scatter);
                gradient::record_operation_with_dims(
                    result.id(),
                    op,
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                    None,
                )?;
            }

            Ok(result)
        }
    }

    pub fn scatter_add<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], Op::Indexing(IndexingOp::ScatterAdd))?;
        validate_same_dtype(&[self, src], Op::Indexing(IndexingOp::ScatterAdd))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::ScatterAdd))?;
        validate_indices_dtype(indices, Op::Indexing(IndexingOp::ScatterAdd))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Indexing(IndexingOp::ScatterAdd));

        let result_layout = self.layout();

        if builder::is_builder_active() {
            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                Op::Indexing(IndexingOp::ScatterAdd),
                Some(OpParams {
                    scalars: vec![dim_scalar],
                    ..Default::default()
                }),
                vec![self.id(), indices.id(), src.id()],
                vec![result_id],
                vec![self.layout(), indices.layout(), src.layout()],
                vec![result_layout],
            )?;

            if requires_grad {
                gradient::record_operation_with_dims(
                    result_id,
                    Op::Indexing(IndexingOp::ScatterAdd),
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                    None,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.call_ops_scatter(
                            &self.layout(),
                            indices_storage,
                            &indices.layout(),
                            src_storage,
                            &src.layout(),
                            dim_usize,
                            Op::Indexing(IndexingOp::ScatterAdd),
                        )
                    })
                })
            })?;

            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Indexing(IndexingOp::ScatterAdd);
                gradient::record_operation_with_dims(
                    result.id(),
                    op,
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                    None,
                )?;
            }

            Ok(result)
        }
    }

    pub fn scatter_max<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], Op::Indexing(IndexingOp::ScatterMax))?;
        validate_same_dtype(&[self, src], Op::Indexing(IndexingOp::ScatterMax))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::ScatterMax))?;
        validate_indices_dtype(indices, Op::Indexing(IndexingOp::ScatterMax))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Indexing(IndexingOp::ScatterMax));

        let result_layout = self.layout();

        if builder::is_builder_active() {
            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                Op::Indexing(IndexingOp::ScatterMax),
                Some(OpParams {
                    scalars: vec![dim_scalar],
                    ..Default::default()
                }),
                vec![self.id(), indices.id(), src.id()],
                vec![result_id],
                vec![self.layout(), indices.layout(), src.layout()],
                vec![result_layout],
            )?;

            if requires_grad {
                gradient::record_operation_with_dims(
                    result_id,
                    Op::Indexing(IndexingOp::ScatterMax),
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                    None,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.call_ops_scatter(
                            &self.layout(),
                            indices_storage,
                            &indices.layout(),
                            src_storage,
                            &src.layout(),
                            dim_usize,
                            Op::Indexing(IndexingOp::ScatterMax),
                        )
                    })
                })
            })?;

            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Indexing(IndexingOp::ScatterMax);
                gradient::record_operation_with_dims(
                    result.id(),
                    op,
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                    None,
                )?;
            }

            Ok(result)
        }
    }

    pub fn scatter_min<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as usize
        } else {
            dim_i32 as usize
        };

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], Op::Indexing(IndexingOp::ScatterMin))?;
        validate_same_dtype(&[self, src], Op::Indexing(IndexingOp::ScatterMin))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Indexing(IndexingOp::ScatterMin))?;
        validate_indices_dtype(indices, Op::Indexing(IndexingOp::ScatterMin))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Indexing(IndexingOp::ScatterMin));

        let result_layout = self.layout();

        if builder::is_builder_active() {
            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                Op::Indexing(IndexingOp::ScatterMin),
                Some(OpParams {
                    scalars: vec![dim_scalar],
                    ..Default::default()
                }),
                vec![self.id(), indices.id(), src.id()],
                vec![result_id],
                vec![self.layout(), indices.layout(), src.layout()],
                vec![result_layout],
            )?;

            if requires_grad {
                gradient::record_operation_with_dims(
                    result_id,
                    Op::Indexing(IndexingOp::ScatterMin),
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                    None,
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.call_ops_scatter(
                            &self.layout(),
                            indices_storage,
                            &indices.layout(),
                            src_storage,
                            &src.layout(),
                            dim_usize,
                            Op::Indexing(IndexingOp::ScatterMin),
                        )
                    })
                })
            })?;

            let requires_grad = (self.is_requires_grad() || src.is_requires_grad()) && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Indexing(IndexingOp::ScatterMin);
                gradient::record_operation_with_dims(
                    result.id(),
                    op,
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                    None,
                )?;
            }

            Ok(result)
        }
    }
}
