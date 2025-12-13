use super::VjpCompute;
use crate::{
    error::HoduResult,
    ops::{LinalgOp, OpParams},
    tensor::TensorId,
};

impl VjpCompute for LinalgOp {
    fn compute_vjp(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _op_params: &OpParams,
    ) -> HoduResult<Vec<TensorId>> {
        match self {
            LinalgOp::Det => {
                // Gradient of determinant:
                // dL/dA = dL/d(det(A)) * det(A) * A^{-T}
                //
                // This requires matrix inverse which is not yet implemented.
                // TODO: Implement when inverse is available
                Err(crate::error::HoduError::UnsupportedOperation(
                    "Gradient for det requires matrix inverse (not yet implemented)".to_string(),
                ))
            },
        }
    }
}
