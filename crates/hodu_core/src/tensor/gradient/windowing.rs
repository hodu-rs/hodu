use super::VjpCompute;
use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::WindowingOp,
    scalar::Scalar,
    tensor::{tensor_from_id, TensorId},
};

impl VjpCompute for WindowingOp {
    fn compute_vjp_with_scalars(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        scalars: &[Scalar],
    ) -> HoduResult<Vec<TensorId>> {
        match self {
            WindowingOp::ReduceWindowMax | WindowingOp::ReduceWindowMin => {
                // Max and Min reductions are not differentiable (discrete operations)
                Err(HoduError::GradientComputationFailed(
                    "Max and Min reductions are not differentiable (discrete operations)".to_string(),
                ))
            },
            WindowingOp::ReduceWindowMean | WindowingOp::ReduceWindowSum => {
                // Parameters: rank, window_shape[rank], strides[rank], padding[rank*2], reduction_type
                if scalars.is_empty() {
                    return Err(HoduError::InternalError("ReduceWindow requires parameters".to_string()));
                }

                let rank = scalars[0].to_usize();
                if scalars.len() < 1 + rank * 4 + 1 {
                    return Err(HoduError::InternalError(
                        "ReduceWindow requires sufficient parameters".to_string(),
                    ));
                }

                // Extract window_shape
                let window_shape: Vec<usize> = (0..rank).map(|i| scalars[1 + i].to_usize()).collect();

                let input = inputs[0];
                let input_tensor = tensor_from_id(input);
                let input_shape = input_tensor.shape();
                let dtype = input_tensor.dtype();

                // For pooling gradient, we need to broadcast the gradient back to input shape
                let grad_tensor = tensor_from_id(grad_output);
                let broadcasted_grad = grad_tensor.broadcast(&input_shape)?;

                match self {
                    WindowingOp::ReduceWindowMean => {
                        // Mean: divide by window size
                        let window_size: usize = window_shape.iter().product();
                        let scale = Scalar::from_f32(1.0 / window_size as f32, dtype);
                        let scaled_grad = broadcasted_grad.mul_scalar(scale)?;
                        Ok(vec![scaled_grad.id()])
                    },
                    WindowingOp::ReduceWindowSum => {
                        // Sum: just broadcast
                        Ok(vec![broadcasted_grad.id()])
                    },
                    _ => unreachable!(),
                }
            },
        }
    }
}
