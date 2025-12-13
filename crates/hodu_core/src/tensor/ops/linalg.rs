use crate::{
    error::HoduResult,
    ops::{DetParams, InvParams, LinalgOp, Op, OpParams},
    tensor::{create_builder_tensor, from_storage_with_context, gradient, Tensor},
    types::{Layout, Shape},
    utils::valid::{validate_dtype_for_device, validate_dtype_for_op, validate_requires_grad_for_op},
};

impl Tensor {
    /// Compute the determinant of a square matrix.
    ///
    /// # Input
    /// - Square matrix `[..., N, N]` (supports batched input)
    ///
    /// # Output
    /// - Scalar `[...]` (batch dimensions preserved)
    ///
    /// # Example
    /// ```ignore
    /// let matrix = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    /// let det = matrix.det()?; // Returns -2.0
    /// ```
    pub fn det(&self) -> HoduResult<Self> {
        let op = Op::Linalg(LinalgOp::Det);

        // Validate dtype for device and operation
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), op.clone())?;

        let shape = self.shape();
        let ndim = shape.ndim();

        // Validate shape - need at least 2D and square matrix
        if ndim < 2 {
            return Err(crate::error::HoduError::InvalidArgument(
                "det requires at least 2D tensor".to_string(),
            ));
        }

        let n = shape.dims()[ndim - 1];
        let m = shape.dims()[ndim - 2];

        if n != m {
            return Err(crate::error::HoduError::InvalidArgument(format!(
                "det requires square matrix, got {}×{}",
                m, n
            )));
        }

        // Compute output shape (batch dimensions only)
        let output_shape = if ndim == 2 {
            Shape::new(&[1])
        } else {
            Shape::new(&shape.dims()[..ndim - 2])
        };

        let result_layout = Layout::from_shape(&output_shape);
        let self_layout = self.layout();
        let validate_requires_grad = validate_requires_grad_for_op(op.clone());

        if crate::snapshot::capture::is_active() {
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                op.clone(),
                Some(OpParams::Det(DetParams)),
                vec![self.id()],
                result_id,
                vec![self_layout],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(vec![self.id()], result_id, op, OpParams::Det(DetParams))?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.call_ops_det(&self_layout))?;

            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(vec![self.id()], result.id(), op, OpParams::Det(DetParams))?;
            }

            Ok(result)
        }
    }

    /// Compute the inverse of a square matrix.
    ///
    /// # Input
    /// - Square matrix `[..., N, N]` (supports batched input)
    ///
    /// # Output
    /// - Inverse matrix `[..., N, N]` (same shape as input)
    ///
    /// # Example
    /// ```ignore
    /// let matrix = Tensor::from_slice(&[4.0, 7.0, 2.0, 6.0], &[2, 2])?;
    /// let inv = matrix.inv()?; // Returns [[0.6, -0.7], [-0.2, 0.4]]
    /// ```
    ///
    /// # Notes
    /// - For singular matrices, the result will contain inf/nan values.
    /// - Uses Gauss-Jordan elimination with partial pivoting for numerical stability.
    pub fn inv(&self) -> HoduResult<Self> {
        let op = Op::Linalg(LinalgOp::Inv);

        // Validate dtype for device and operation
        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), op.clone())?;

        let shape = self.shape();
        let ndim = shape.ndim();

        // Validate shape - need at least 2D and square matrix
        if ndim < 2 {
            return Err(crate::error::HoduError::InvalidArgument(
                "inv requires at least 2D tensor".to_string(),
            ));
        }

        let n = shape.dims()[ndim - 1];
        let m = shape.dims()[ndim - 2];

        if n != m {
            return Err(crate::error::HoduError::InvalidArgument(format!(
                "inv requires square matrix, got {}×{}",
                m, n
            )));
        }

        // Output shape is same as input
        let output_shape = shape.clone();
        let result_layout = Layout::from_shape(&output_shape);
        let self_layout = self.layout();
        let validate_requires_grad = validate_requires_grad_for_op(op.clone());

        if crate::snapshot::capture::is_active() {
            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                op.clone(),
                Some(OpParams::Inv(InvParams)),
                vec![self.id()],
                result_id,
                vec![self_layout],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(vec![self.id()], result_id, op, OpParams::Inv(InvParams))?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|storage| storage.call_ops_inv(&self_layout))?;

            let requires_grad = self.is_requires_grad() && validate_requires_grad;
            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(vec![self.id()], result.id(), op, OpParams::Inv(InvParams))?;
            }

            Ok(result)
        }
    }
}
