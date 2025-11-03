use crate::{
    builder,
    compat::*,
    error::HoduResult,
    op::{
        self,
        utils::{
            validate_dtype_for_device, validate_dtype_for_op, validate_indices_dtype, validate_same_device,
            validate_same_dtype,
        },
        Op,
    },
    scalar::Scalar,
    tensor::{
        create_builder_tensor_with_grad, from_storage_with_grad, gradient, register_operation_in_builder, Tensor,
    },
    types::layout::Layout,
};

// Indexing Operations
impl Tensor {
    pub fn index_select<D: Into<Scalar>>(&self, dim: D, indices: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices], "index_select")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "index_select")?;
        let op = Op::Indexing(
            op::IndexingOp::IndexSelect,
            vec![self.id(), indices.id()],
            vec![dim_scalar],
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_indices_dtype(indices, "index_select")?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let indices_layout = indices.get_layout();
            let shape = layout.get_shape();

            // Output shape: replace indexed dimension with indices size
            let mut output_shape = shape.to_vec();
            output_shape[dim_usize] = indices_layout.get_size();

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![layout.clone(), indices_layout.clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), indices.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    storage.index_select(&self.get_layout(), indices_storage, &indices.get_layout(), dim_usize)
                })
            })?;

            let layout = self.get_layout();
            let indices_layout = indices.get_layout();
            let shape = layout.get_shape();
            let mut output_shape = shape.to_vec();
            output_shape[dim_usize] = indices_layout.get_size();

            let result_layout = Layout::from_shape(&output_shape);
            let requires_grad = self.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Indexing(
                    op::IndexingOp::IndexSelect,
                    vec![self.id(), indices.id()],
                    vec![dim_scalar],
                );
                gradient::record_operation(result.id(), op, vec![self.id(), indices.id()])?;
            }

            Ok(result)
        }
    }

    pub fn index_put<D: Into<Scalar>>(&self, dim: D, indices: &Self, values: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, values], "index_put")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "index_put")?;
        let op = Op::Indexing(
            op::IndexingOp::IndexPut,
            vec![self.id(), indices.id(), values.id()],
            vec![dim_scalar],
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, values], "index_put")?;
        validate_indices_dtype(indices, "index_put")?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let indices_layout = indices.get_layout();
            let values_layout = values.get_layout();

            // Output has same shape as input
            let result_layout = layout.clone();
            let requires_grad = self.is_requires_grad() || values.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![layout.clone(), indices_layout.clone(), values_layout.clone()],
                vec![result_layout],
            );

            if requires_grad {
                gradient::record_operation(result_id, op, vec![self.id(), values.id(), indices.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    values.with_storage(|values_storage| {
                        storage.index_put(
                            &self.get_layout(),
                            indices_storage,
                            &indices.get_layout(),
                            values_storage,
                            &values.get_layout(),
                            dim_usize,
                        )
                    })
                })
            })?;

            let result_layout = self.get_layout();
            let requires_grad = self.is_requires_grad() || values.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                let op = Op::Indexing(
                    op::IndexingOp::IndexPut,
                    vec![self.id(), indices.id(), values.id()],
                    vec![dim_scalar],
                );
                gradient::record_operation(result.id(), op, vec![self.id(), values.id(), indices.id()])?;
            }

            Ok(result)
        }
    }

    pub fn gather<D: Into<Scalar>>(&self, dim: D, indices: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices], "gather")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "gather")?;
        let op = Op::Indexing(op::IndexingOp::Gather, vec![self.id(), indices.id()], vec![dim_scalar]);
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_indices_dtype(indices, "gather")?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let indices_layout = indices.get_layout();

            // Output has same shape as indices
            let result_layout = indices_layout.clone();
            let requires_grad = self.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![layout.clone(), indices_layout.clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), indices.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    storage.gather(&self.get_layout(), indices_storage, &indices.get_layout(), dim_usize)
                })
            })?;

            let result_layout = indices.get_layout().clone();
            let requires_grad = self.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && self.is_requires_grad() {
                let op = Op::Indexing(op::IndexingOp::Gather, vec![self.id(), indices.id()], vec![dim_scalar]);
                gradient::record_operation(result.id(), op, vec![self.id(), indices.id()])?;
            }

            Ok(result)
        }
    }

    pub fn scatter<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], "scatter")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "scatter")?;
        let op = Op::Indexing(
            op::IndexingOp::Scatter,
            vec![self.id(), indices.id(), src.id()],
            vec![dim_scalar],
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, src], "scatter")?;
        validate_indices_dtype(indices, "scatter")?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let indices_layout = indices.get_layout();
            let src_layout = src.get_layout();

            // Output has same shape as self
            let result_layout = layout.clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![layout.clone(), indices_layout.clone(), src_layout.clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() || src.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.scatter(
                            &self.get_layout(),
                            indices_storage,
                            &indices.get_layout(),
                            src_storage,
                            &src.get_layout(),
                            dim_usize,
                        )
                    })
                })
            })?;

            let result_layout = self.get_layout().clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && (self.is_requires_grad() || src.is_requires_grad()) {
                let op = Op::Indexing(
                    op::IndexingOp::Scatter,
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                );
                gradient::record_operation(result.id(), op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result)
        }
    }

    pub fn scatter_add<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], "scatter_add")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "scatter_add")?;
        let op = Op::Indexing(
            op::IndexingOp::ScatterAdd,
            vec![self.id(), indices.id(), src.id()],
            vec![dim_scalar],
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, src], "scatter_add")?;
        validate_indices_dtype(indices, "scatter_add")?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let indices_layout = indices.get_layout();
            let src_layout = src.get_layout();

            let result_layout = layout.clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![layout.clone(), indices_layout.clone(), src_layout.clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() || src.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.scatter_add(
                            &self.get_layout(),
                            indices_storage,
                            &indices.get_layout(),
                            src_storage,
                            &src.get_layout(),
                            dim_usize,
                        )
                    })
                })
            })?;

            let result_layout = self.get_layout().clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && (self.is_requires_grad() || src.is_requires_grad()) {
                let op = Op::Indexing(
                    op::IndexingOp::ScatterAdd,
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                );
                gradient::record_operation(result.id(), op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result)
        }
    }

    pub fn scatter_max<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], "scatter_max")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "scatter_max")?;
        let op = Op::Indexing(
            op::IndexingOp::ScatterMax,
            vec![self.id(), indices.id(), src.id()],
            vec![dim_scalar],
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, src], "scatter_max")?;
        validate_indices_dtype(indices, "scatter_max")?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let indices_layout = indices.get_layout();
            let src_layout = src.get_layout();

            let result_layout = layout.clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![layout.clone(), indices_layout.clone(), src_layout.clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() || src.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.scatter_max(
                            &self.get_layout(),
                            indices_storage,
                            &indices.get_layout(),
                            src_storage,
                            &src.get_layout(),
                            dim_usize,
                        )
                    })
                })
            })?;

            let result_layout = self.get_layout().clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && (self.is_requires_grad() || src.is_requires_grad()) {
                let op = Op::Indexing(
                    op::IndexingOp::ScatterMax,
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                );
                gradient::record_operation(result.id(), op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result)
        }
    }

    pub fn scatter_min<D: Into<Scalar>>(&self, dim: D, indices: &Self, src: &Self) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, indices, src], "scatter_min")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "scatter_min")?;
        let op = Op::Indexing(
            op::IndexingOp::ScatterMin,
            vec![self.id(), indices.id(), src.id()],
            vec![dim_scalar],
        );
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, src], "scatter_min")?;
        validate_indices_dtype(indices, "scatter_min")?;

        if builder::is_builder_active() {
            let layout = self.get_layout();
            let indices_layout = indices.get_layout();
            let src_layout = src.get_layout();

            let result_layout = layout.clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![layout.clone(), indices_layout.clone(), src_layout.clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() || src.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| {
                indices.with_storage(|indices_storage| {
                    src.with_storage(|src_storage| {
                        storage.scatter_min(
                            &self.get_layout(),
                            indices_storage,
                            &indices.get_layout(),
                            src_storage,
                            &src.get_layout(),
                            dim_usize,
                        )
                    })
                })
            })?;

            let result_layout = self.get_layout().clone();
            let requires_grad = self.is_requires_grad() || src.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && (self.is_requires_grad() || src.is_requires_grad()) {
                let op = Op::Indexing(
                    op::IndexingOp::ScatterMin,
                    vec![self.id(), indices.id(), src.id()],
                    vec![dim_scalar],
                );
                gradient::record_operation(result.id(), op, vec![self.id(), src.id(), indices.id()])?;
            }

            Ok(result)
        }
    }
}
