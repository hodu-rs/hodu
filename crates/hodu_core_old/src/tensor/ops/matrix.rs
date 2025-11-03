use crate::{
    builder,
    compat::*,
    error::{HoduError, HoduResult},
    op::{
        self,
        utils::{validate_dtype_for_device, validate_dtype_for_op, validate_same_device, validate_same_dtype},
        Op,
    },
    tensor::{
        create_builder_tensor_with_grad, from_storage_with_grad, gradient, register_operation_in_builder, Tensor,
    },
    types::layout::Layout,
};

// Matrix operations
impl Tensor {
    pub fn matmul(&self, other: &Self) -> HoduResult<Self> {
        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, other], "matmul")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "matmul")?;
        let op = Op::Matrix(op::MatrixOp::Matmul, self.id(), other.id());
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, other], "matmul")?;

        // Supports ND batched matrix multiplication with broadcasting (like XLA matmul)
        let lhs_layout = self.get_layout();
        let rhs_layout = other.get_layout();
        let lhs_shape = lhs_layout.get_shape();
        let rhs_shape = rhs_layout.get_shape();
        let lhs_ndim = lhs_shape.len();
        let rhs_ndim = rhs_shape.len();

        // Both tensors must be at least 1D
        if lhs_ndim < 1 || rhs_ndim < 1 {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
                op: "matmul - both tensors must be at least 1D".to_string(),
            });
        }

        // Handle 1D x 1D case - vector dot product
        if lhs_ndim == 1 && rhs_ndim == 1 {
            if lhs_shape[0] != rhs_shape[0] {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "matmul - 1D vectors must have same length".to_string(),
                });
            }
            return self.mul(other)?.sum_all();
        }

        // Handle 2D x 1D case - matrix-vector product
        if lhs_ndim == 2 && rhs_ndim == 1 {
            let k1 = lhs_shape[1];
            let k2 = rhs_shape[0];

            if k1 != k2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "matmul - incompatible dimensions for matrix-vector product".to_string(),
                });
            }

            let rhs_reshaped = other.reshape([k2, 1])?;
            let result = self.dot_2d(&rhs_reshaped)?;
            return result.squeeze(Some(-1));
        }

        // Handle 1D x 2D case - vector-matrix product
        if lhs_ndim == 1 && rhs_ndim == 2 {
            let k1 = lhs_shape[0];
            let k2 = rhs_shape[0];

            if k1 != k2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "matmul - incompatible dimensions for vector-matrix product".to_string(),
                });
            }

            let lhs_reshaped = self.reshape([1, k1])?;
            let result = lhs_reshaped.dot_2d(other)?;
            return result.squeeze(Some(0));
        }

        // Handle 2D x 2D case - standard matrix multiplication
        if lhs_ndim == 2 && rhs_ndim == 2 {
            return self.dot_2d(other);
        }

        // Handle ND x ND case - batched matrix multiplication with broadcasting
        self.matmul_batched(other)
    }

    // Helper: ND batched matrix multiplication
    fn matmul_batched(&self, other: &Self) -> HoduResult<Self> {
        let lhs_layout = self.get_layout();
        let rhs_layout = other.get_layout();
        let lhs_shape = lhs_layout.get_shape();
        let rhs_shape = rhs_layout.get_shape();
        let lhs_ndim = lhs_shape.len();
        let rhs_ndim = rhs_shape.len();

        if lhs_ndim < 2 || rhs_ndim < 2 {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
                op: "matmul_batched - both tensors must be at least 2D".to_string(),
            });
        }

        // Check that last two dimensions are compatible for matmul
        let lhs_inner = lhs_shape[lhs_ndim - 1];
        let rhs_outer = rhs_shape[rhs_ndim - 2];

        if lhs_inner != rhs_outer {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
                op: "matmul_batched - inner dimensions must match".to_string(),
            });
        }

        // Compute broadcast shape for batch dimensions
        let lhs_batch_shape = &lhs_shape[..lhs_ndim - 2];
        let rhs_batch_shape = &rhs_shape[..rhs_ndim - 2];

        let max_batch_ndim = lhs_batch_shape.len().max(rhs_batch_shape.len());
        let mut batch_shape = vec![0; max_batch_ndim];

        for i in 0..max_batch_ndim {
            let lhs_dim = if i < lhs_batch_shape.len() {
                lhs_batch_shape[lhs_batch_shape.len() - 1 - i]
            } else {
                1
            };
            let rhs_dim = if i < rhs_batch_shape.len() {
                rhs_batch_shape[rhs_batch_shape.len() - 1 - i]
            } else {
                1
            };

            if lhs_dim != 1 && rhs_dim != 1 && lhs_dim != rhs_dim {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "matmul_batched - incompatible batch dimensions".to_string(),
                });
            }
            batch_shape[max_batch_ndim - 1 - i] = lhs_dim.max(rhs_dim);
        }

        // Broadcast both tensors to have the same batch dimensions
        let mut lhs_broadcast_shape = batch_shape.clone();
        lhs_broadcast_shape.push(lhs_shape[lhs_ndim - 2]);
        lhs_broadcast_shape.push(lhs_shape[lhs_ndim - 1]);

        let mut rhs_broadcast_shape = batch_shape.clone();
        rhs_broadcast_shape.push(rhs_shape[rhs_ndim - 2]);
        rhs_broadcast_shape.push(rhs_shape[rhs_ndim - 1]);

        let lhs_broadcasted = self.broadcast(&lhs_broadcast_shape)?;
        let rhs_broadcasted = other.broadcast(&rhs_broadcast_shape)?;

        // Result shape: batch_shape + [M, N]
        let mut result_shape = batch_shape;
        result_shape.push(lhs_shape[lhs_ndim - 2]); // M
        result_shape.push(rhs_shape[rhs_ndim - 1]); // N

        if builder::is_builder_active() {
            let result_layout = Layout::from_shape(&result_shape);
            let requires_grad = self.is_requires_grad() || other.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let op = Op::Matrix(op::MatrixOp::Matmul, lhs_broadcasted.id(), rhs_broadcasted.id());
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![
                    lhs_broadcasted.get_layout().clone(),
                    rhs_broadcasted.get_layout().clone(),
                ],
                vec![result_layout],
            );

            if self.is_requires_grad() || other.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), other.id()])?;
            }

            Ok(result_tensor)
        } else {
            let result_layout = Layout::from_shape(&result_shape);

            let storage = lhs_broadcasted.with_storage(|lhs_storage| {
                rhs_broadcasted.with_storage(|rhs_storage| {
                    lhs_storage.matmul(
                        rhs_storage,
                        &lhs_broadcasted.get_layout(),
                        &rhs_broadcasted.get_layout(),
                    )
                })
            })?;

            let requires_grad = self.is_requires_grad() || other.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && (self.is_requires_grad() || other.is_requires_grad()) {
                let op = Op::Matrix(op::MatrixOp::Matmul, lhs_broadcasted.id(), rhs_broadcasted.id());
                gradient::record_operation(result.id(), op, vec![self.id(), other.id()])?;
            }

            Ok(result)
        }
    }

    pub fn dot(&self, other: &Self) -> HoduResult<Self> {
        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, other], "dot")?;
        validate_dtype_for_device(self.get_dtype(), &self.get_device(), "dot")?;
        let op = Op::Matrix(op::MatrixOp::Dot, self.id(), other.id());
        validate_dtype_for_op(self.get_dtype(), &op)?;
        validate_same_dtype(&[self, other], "dot")?;

        // Simple dot operation - supports 1D and 2D combinations (like XLA dot)
        let lhs_layout = self.get_layout();
        let rhs_layout = other.get_layout();
        let lhs_shape = lhs_layout.get_shape();
        let rhs_shape = rhs_layout.get_shape();
        let lhs_ndim = lhs_shape.len();
        let rhs_ndim = rhs_shape.len();

        // Case 1: 1D x 1D - dot product (inner product)
        if lhs_ndim == 1 && rhs_ndim == 1 {
            if lhs_shape[0] != rhs_shape[0] {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "dot - 1D vectors must have same length".to_string(),
                });
            }
            return self.mul(other)?.sum_all();
        }

        // Case 2: 2D x 1D - matrix-vector product
        if lhs_ndim == 2 && rhs_ndim == 1 {
            let k1 = lhs_shape[1];
            let k2 = rhs_shape[0];

            if k1 != k2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "dot - incompatible dimensions for matrix-vector product".to_string(),
                });
            }

            let rhs_reshaped = other.reshape([k2, 1])?;
            let result = self.dot_2d(&rhs_reshaped)?;
            return result.squeeze(Some(-1));
        }

        // Case 3: 1D x 2D - vector-matrix product
        if lhs_ndim == 1 && rhs_ndim == 2 {
            let k1 = lhs_shape[0];
            let k2 = rhs_shape[0];

            if k1 != k2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "dot - incompatible dimensions for vector-matrix product".to_string(),
                });
            }

            let lhs_reshaped = self.reshape([1, k1])?;
            let result = lhs_reshaped.dot_2d(other)?;
            return result.squeeze(Some(0));
        }

        // Case 4: 2D x 2D - matrix multiplication
        if lhs_ndim == 2 && rhs_ndim == 2 {
            return self.dot_2d(other);
        }

        // Dot operation only supports 1D and 2D tensors
        // For higher dimensional (batched) operations, use matmul instead
        Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
            op: "dot - only supports 1D and 2D tensors. Use matmul() for batched operations".to_string(),
        })
    }

    // Helper: 2D matrix multiplication (assumes both are 2D)
    fn dot_2d(&self, other: &Self) -> HoduResult<Self> {
        if builder::is_builder_active() {
            let lhs_layout = self.get_layout();
            let rhs_layout = other.get_layout();
            let lhs_shape = lhs_layout.get_shape();
            let rhs_shape = rhs_layout.get_shape();

            let (m, k1) = (lhs_shape[0], lhs_shape[1]);
            let (k2, n) = (rhs_shape[0], rhs_shape[1]);

            if k1 != k2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape.to_vec(),
                    rhs: rhs_shape.to_vec(),
                    op: "dot_2d - inner dimensions must match".to_string(),
                });
            }

            let result_shape = vec![m, n];
            let result_layout = Layout::from_shape(&result_shape);
            let requires_grad = self.is_requires_grad() || other.is_requires_grad();
            let (result_id, result_tensor) = create_builder_tensor_with_grad(result_layout.clone(), requires_grad);

            let op = Op::Matrix(op::MatrixOp::Dot, self.id(), other.id());
            register_operation_in_builder(
                op.clone(),
                vec![result_id],
                vec![self.get_layout().clone(), other.get_layout().clone()],
                vec![result_layout],
            );

            if self.is_requires_grad() || other.is_requires_grad() {
                gradient::record_operation(result_id, op, vec![self.id(), other.id()])?;
            }

            Ok(result_tensor)
        } else {
            let lhs_layout = self.get_layout();
            let rhs_layout = other.get_layout();
            let lhs_shape = lhs_layout.get_shape();
            let rhs_shape = rhs_layout.get_shape();

            let (m, _k1) = (lhs_shape[0], lhs_shape[1]);
            let (_k2, n) = (rhs_shape[0], rhs_shape[1]);
            let result_shape = vec![m, n];
            let result_layout = Layout::from_shape(&result_shape);

            let storage = self.with_storage(|lhs_storage| {
                other.with_storage(|rhs_storage| lhs_storage.dot(rhs_storage, &self.get_layout(), &other.get_layout()))
            })?;

            let requires_grad = self.is_requires_grad() || other.is_requires_grad();
            let result = from_storage_with_grad(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && (self.is_requires_grad() || other.is_requires_grad()) {
                let op = Op::Matrix(op::MatrixOp::Dot, self.id(), other.id());
                gradient::record_operation(result.id(), op, vec![self.id(), other.id()])?;
            }

            Ok(result)
        }
    }
}
