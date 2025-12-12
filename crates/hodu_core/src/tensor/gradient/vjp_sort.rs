use super::VjpCompute;
use crate::{
    error::{HoduError, HoduResult},
    op_params::{OpParams, TopKParams},
    ops::SortOp,
    tensor::{tensor_from_id, Tensor, TensorId},
};

impl VjpCompute for SortOp {
    fn compute_vjp(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        op_params: &OpParams,
    ) -> HoduResult<Vec<TensorId>> {
        match self {
            SortOp::TopK => {
                let OpParams::TopK(TopKParams { dim, indices_id, .. }) = op_params else {
                    return Err(HoduError::VjpFunctionNotFound("TopK requires TopKParams".to_string()));
                };

                if inputs.is_empty() {
                    return Err(HoduError::InternalError("TopK requires 1 input".to_string()));
                }

                let input_id = inputs[0];
                let input_tensor = tensor_from_id(input_id);
                let input_shape = input_tensor.shape();
                let dtype = input_tensor.dtype();

                // Create zero tensor with same shape as input
                let grad_input = Tensor::zeros(&input_shape, dtype)?;

                // Scatter the gradient back to the original positions using indices
                let indices_tensor = tensor_from_id(*indices_id);
                let grad_tensor = tensor_from_id(grad_output);

                // Normalize dim
                let ndim = input_shape.ndim();
                let dim_normalized = if *dim < 0 {
                    (ndim as i32 + dim) as usize
                } else {
                    *dim as usize
                };

                let result = grad_input.scatter_add(dim_normalized, &indices_tensor, &grad_tensor)?;

                Ok(vec![result.id()])
            },
        }
    }
}
