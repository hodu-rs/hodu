use crate::{
    capture,
    compat::*,
    error::{HoduError, HoduResult},
    ops::{DotParams, MatmulParams, MatrixOp, Op, OpParams},
    tensor::{create_builder_tensor, from_storage_with_context, gradient, Tensor},
    types::{Layout, Shape},
    utils::valid::{
        validate_dtype_for_device, validate_dtype_for_op, validate_requires_grad_for_op, validate_same_device,
        validate_same_dtype,
    },
};

impl Tensor {
    pub fn matmul(&self, other: &Self) -> HoduResult<Self> {
        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, other], Op::Matrix(MatrixOp::Matmul))?;
        validate_same_dtype(&[self, other], Op::Matrix(MatrixOp::Matmul))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Matrix(MatrixOp::Matmul))?;

        // Supports ND batched matrix multiplication with broadcasting (like XLA matmul)
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        let lhs_dims = lhs_shape.dims();
        let rhs_dims = rhs_shape.dims();
        let lhs_ndim = lhs_dims.len();
        let rhs_ndim = rhs_dims.len();

        // Both tensors must be at least 1D
        if lhs_ndim < 1 || rhs_ndim < 1 {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape,
                rhs: rhs_shape,
                op: Op::Matrix(MatrixOp::Matmul),
            });
        }

        // Handle 1D x 1D case - vector dot product
        if lhs_ndim == 1 && rhs_ndim == 1 {
            if lhs_dims[0] != rhs_dims[0] {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape,
                    rhs: rhs_shape,
                    op: Op::Matrix(MatrixOp::Matmul),
                });
            }
            return self.mul(other)?.sum_all();
        }

        // Handle 2D x 1D case - matrix-vector product
        if lhs_ndim == 2 && rhs_ndim == 1 {
            let k1 = lhs_dims[1];
            let k2 = rhs_dims[0];

            if k1 != k2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape,
                    rhs: rhs_shape,
                    op: Op::Matrix(MatrixOp::Matmul),
                });
            }

            let rhs_reshaped = other.reshape(Shape::from(vec![k2, 1]))?;
            let result = self.dot_2d(&rhs_reshaped)?;
            return result.squeeze(&[-1]);
        }

        // Handle 1D x 2D case - vector-matrix product
        if lhs_ndim == 1 && rhs_ndim == 2 {
            let k1 = lhs_dims[0];
            let k2 = rhs_dims[0];

            if k1 != k2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape,
                    rhs: rhs_shape,
                    op: Op::Matrix(MatrixOp::Matmul),
                });
            }

            let lhs_reshaped = self.reshape(Shape::from(vec![1, k1]))?;
            let result = lhs_reshaped.dot_2d(other)?;
            return result.squeeze(&[0]);
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
        let validate_requires_grad = validate_requires_grad_for_op(Op::Matrix(MatrixOp::Matmul));

        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        let lhs_dims = lhs_shape.dims();
        let rhs_dims = rhs_shape.dims();
        let lhs_ndim = lhs_dims.len();
        let rhs_ndim = rhs_dims.len();

        if lhs_ndim < 2 || rhs_ndim < 2 {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape,
                rhs: rhs_shape,
                op: Op::Matrix(MatrixOp::Matmul),
            });
        }

        // Check that last two dimensions are compatible for matmul
        let lhs_inner = lhs_dims[lhs_ndim - 1];
        let rhs_outer = rhs_dims[rhs_ndim - 2];

        if lhs_inner != rhs_outer {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape,
                rhs: rhs_shape,
                op: Op::Matrix(MatrixOp::Matmul),
            });
        }

        // Compute broadcast shape for batch dimensions
        let lhs_batch_dims = &lhs_dims[..lhs_ndim - 2];
        let rhs_batch_dims = &rhs_dims[..rhs_ndim - 2];

        let max_batch_ndim = lhs_batch_dims.len().max(rhs_batch_dims.len());
        let mut batch_dims = vec![0; max_batch_ndim];

        for i in 0..max_batch_ndim {
            let lhs_dim = if i < lhs_batch_dims.len() {
                lhs_batch_dims[lhs_batch_dims.len() - 1 - i]
            } else {
                1
            };
            let rhs_dim = if i < rhs_batch_dims.len() {
                rhs_batch_dims[rhs_batch_dims.len() - 1 - i]
            } else {
                1
            };

            if lhs_dim != 1 && rhs_dim != 1 && lhs_dim != rhs_dim {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape,
                    rhs: rhs_shape,
                    op: Op::Matrix(MatrixOp::Matmul),
                });
            }
            batch_dims[max_batch_ndim - 1 - i] = lhs_dim.max(rhs_dim);
        }

        // Broadcast both tensors to have the same batch dimensions
        // Pre-allocate with exact capacity to avoid reallocation
        let mut lhs_broadcast_dims = Vec::with_capacity(max_batch_ndim + 2);
        lhs_broadcast_dims.extend_from_slice(&batch_dims);
        lhs_broadcast_dims.push(lhs_dims[lhs_ndim - 2]);
        lhs_broadcast_dims.push(lhs_dims[lhs_ndim - 1]);

        let mut rhs_broadcast_dims = Vec::with_capacity(max_batch_ndim + 2);
        rhs_broadcast_dims.extend_from_slice(&batch_dims);
        rhs_broadcast_dims.push(rhs_dims[rhs_ndim - 2]);
        rhs_broadcast_dims.push(rhs_dims[rhs_ndim - 1]);

        let lhs_broadcasted = self.broadcast(Shape::from(lhs_broadcast_dims))?;
        let rhs_broadcasted = other.broadcast(Shape::from(rhs_broadcast_dims))?;

        // Result shape: batch_dims + [M, N]
        // Reuse batch_dims vec instead of cloning
        let mut result_dims = batch_dims;
        result_dims.push(lhs_dims[lhs_ndim - 2]); // M
        result_dims.push(rhs_dims[rhs_ndim - 1]); // N

        let result_layout = Layout::from_shape(&Shape::from(result_dims));
        let lhs_layout = lhs_broadcasted.layout();
        let rhs_layout = rhs_broadcasted.layout();

        if capture::is_active() {
            let requires_grad = (self.is_requires_grad() || other.is_requires_grad()) && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), requires_grad);

            capture::capture_operation(
                Op::Matrix(MatrixOp::Matmul),
                Some(OpParams::Matmul(MatmulParams)),
                vec![lhs_broadcasted.id(), rhs_broadcasted.id()],
                result_id,
                vec![lhs_layout, rhs_layout],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![lhs_broadcasted.id(), rhs_broadcasted.id()],
                    result_id,
                    Op::Matrix(MatrixOp::Matmul),
                    OpParams::Matmul(MatmulParams),
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = lhs_broadcasted.with_storage(|lhs_storage| {
                rhs_broadcasted.with_storage(|rhs_storage| {
                    lhs_storage.call_ops_matmul(rhs_storage, &lhs_layout, &rhs_layout, Op::Matrix(MatrixOp::Matmul))
                })
            })?;

            let requires_grad = self.is_requires_grad() || other.is_requires_grad();
            let requires_grad = requires_grad && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![lhs_broadcasted.id(), rhs_broadcasted.id()],
                    result.id(),
                    Op::Matrix(MatrixOp::Matmul),
                    OpParams::Matmul(MatmulParams),
                )?;
            }

            Ok(result)
        }
    }

    pub fn dot(&self, other: &Self) -> HoduResult<Self> {
        // Validate device, dtype for device, and dtype for operation
        validate_same_device(&[self, other], Op::Matrix(MatrixOp::Dot))?;
        validate_same_dtype(&[self, other], Op::Matrix(MatrixOp::Dot))?;
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Matrix(MatrixOp::Dot))?;

        // Simple dot operation - supports 1D and 2D combinations (like XLA dot)
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        let lhs_dims = lhs_shape.dims();
        let rhs_dims = rhs_shape.dims();
        let lhs_ndim = lhs_dims.len();
        let rhs_ndim = rhs_dims.len();

        // Case 1: 1D x 1D - dot product (inner product)
        if lhs_ndim == 1 && rhs_ndim == 1 {
            if lhs_dims[0] != rhs_dims[0] {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape,
                    rhs: rhs_shape,
                    op: Op::Matrix(MatrixOp::Dot),
                });
            }
            return self.mul(other)?.sum_all();
        }

        // Case 2: 2D x 1D - matrix-vector product
        if lhs_ndim == 2 && rhs_ndim == 1 {
            let k1 = lhs_dims[1];
            let k2 = rhs_dims[0];

            if k1 != k2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape,
                    rhs: rhs_shape,
                    op: Op::Matrix(MatrixOp::Dot),
                });
            }

            let rhs_reshaped = other.reshape(Shape::from(vec![k2, 1]))?;
            let result = self.dot_2d(&rhs_reshaped)?;
            return result.squeeze(&[-1]);
        }

        // Case 3: 1D x 2D - vector-matrix product
        if lhs_ndim == 1 && rhs_ndim == 2 {
            let k1 = lhs_dims[0];
            let k2 = rhs_dims[0];

            if k1 != k2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: lhs_shape,
                    rhs: rhs_shape,
                    op: Op::Matrix(MatrixOp::Dot),
                });
            }

            let lhs_reshaped = self.reshape(Shape::from(vec![1, k1]))?;
            let result = lhs_reshaped.dot_2d(other)?;
            return result.squeeze(&[0]);
        }

        // Case 4: 2D x 2D - matrix multiplication
        if lhs_ndim == 2 && rhs_ndim == 2 {
            return self.dot_2d(other);
        }

        // Dot operation only supports 1D and 2D tensors
        // For higher dimensional (batched) operations, use matmul instead
        Err(HoduError::InternalError(
            "dot - only supports 1D and 2D tensors. Use matmul() for batched operations".to_string(),
        ))
    }

    // Helper: 2D matrix multiplication (assumes both are 2D)
    fn dot_2d(&self, other: &Self) -> HoduResult<Self> {
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        let lhs_dims = lhs_shape.dims();
        let rhs_dims = rhs_shape.dims();

        let (m, k1) = (lhs_dims[0], lhs_dims[1]);
        let (k2, n) = (rhs_dims[0], rhs_dims[1]);

        if k1 != k2 {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape,
                rhs: rhs_shape,
                op: Op::Matrix(MatrixOp::Dot),
            });
        }

        let result_dims = vec![m, n];
        let result_layout = Layout::from_shape(&Shape::from(result_dims));
        let validate_requires_grad = validate_requires_grad_for_op(Op::Matrix(MatrixOp::Dot));

        let self_layout = self.layout();
        let other_layout = other.layout();

        if capture::is_active() {
            let requires_grad = (self.is_requires_grad() || other.is_requires_grad()) && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), requires_grad);

            capture::capture_operation(
                Op::Matrix(MatrixOp::Dot),
                Some(OpParams::Dot(DotParams)),
                vec![self.id(), other.id()],
                result_id,
                vec![self_layout, other_layout],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(
                    vec![self.id(), other.id()],
                    result_id,
                    Op::Matrix(MatrixOp::Dot),
                    OpParams::Dot(DotParams),
                )?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|lhs_storage| {
                other.with_storage(|rhs_storage| {
                    lhs_storage.call_ops_dot(rhs_storage, &self_layout, &other_layout, Op::Matrix(MatrixOp::Dot))
                })
            })?;

            let requires_grad = self.is_requires_grad() || other.is_requires_grad();
            let requires_grad = requires_grad && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(
                    vec![self.id(), other.id()],
                    result.id(),
                    Op::Matrix(MatrixOp::Dot),
                    OpParams::Dot(DotParams),
                )?;
            }

            Ok(result)
        }
    }
}
