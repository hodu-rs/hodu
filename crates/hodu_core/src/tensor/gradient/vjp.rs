use crate::{
    error::{HoduError, HoduResult},
    ops::OpParams,
    tensor::TensorId,
};

/// VJP (Vector-Jacobian Product) computation trait
///
/// All operations that support gradient computation implement this trait.
/// VJP computes the gradient of outputs with respect to inputs given
/// the gradient of the loss with respect to outputs.
pub(crate) trait VjpCompute {
    fn compute_vjp(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _op_params: &OpParams,
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(
            "compute_vjp not implemented".to_string(),
        ))
    }
}
