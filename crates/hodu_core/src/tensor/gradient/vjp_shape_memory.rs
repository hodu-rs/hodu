use super::VjpCompute;
use crate::{
    error::{HoduError, HoduResult},
    op_params::{FlipParams, OpParams},
    ops::ShapeMemoryOp,
    tensor::{tensor_from_id, TensorId},
};

impl VjpCompute for ShapeMemoryOp {
    fn compute_vjp(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        op_params: &OpParams,
    ) -> HoduResult<Vec<TensorId>> {
        match self {
            ShapeMemoryOp::Flip => {
                let OpParams::Flip(FlipParams { dims }) = op_params else {
                    return Err(HoduError::VjpFunctionNotFound("Flip requires FlipParams".to_string()));
                };

                let grad_tensor = tensor_from_id(grad_output);

                // Gradient of flip is just flip along the same dimensions
                // d/dx flip(x, dims) = flip(grad, dims)
                let grad_input = grad_tensor.flip(dims)?;

                Ok(vec![grad_input.id()])
            },
        }
    }
}
