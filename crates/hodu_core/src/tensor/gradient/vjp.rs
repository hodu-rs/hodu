use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    scalar::Scalar,
    tensor::TensorId,
};

/// VJP (Vector-Jacobian Product) computation trait
///
/// All operations that support gradient computation implement this trait.
/// VJP computes the gradient of outputs with respect to inputs given
/// the gradient of the loss with respect to outputs.
pub(crate) trait VjpCompute {
    /// Compute VJP for operations without additional parameters
    fn compute_vjp(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(
            "compute_vjp not implemented".to_string(),
        ))
    }

    /// Compute VJP for operations with a scalar parameter
    fn compute_vjp_with_scalar(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _scalar: Scalar,
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(
            "compute_vjp_with_scalar not implemented".to_string(),
        ))
    }

    /// Compute VJP for operations with multiple scalar parameters
    fn compute_vjp_with_scalars(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _scalars: &[Scalar],
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(
            "compute_vjp_with_scalars not implemented".to_string(),
        ))
    }

    /// Compute VJP for operations with dimension parameters
    fn compute_vjp_with_dims(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _dims: &[Scalar],
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(
            "compute_vjp_with_dims not implemented".to_string(),
        ))
    }

    /// Compute VJP for split operations with output index
    fn compute_vjp_with_split_info(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _params: &[Scalar],
        _output_index: usize,
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(
            "compute_vjp_with_split_info not implemented".to_string(),
        ))
    }
}
