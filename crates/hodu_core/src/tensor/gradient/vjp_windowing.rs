use super::VjpCompute;
use crate::{
    error::{HoduError, HoduResult},
    ops::{OpParams, ReduceWindowParams, WindowingOp},
    scalar::Scalar,
    tensor::{tensor_from_id, TensorId},
};

impl VjpCompute for WindowingOp {
    fn compute_vjp(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        op_params: &OpParams,
    ) -> HoduResult<Vec<TensorId>> {
        let OpParams::ReduceWindow(ReduceWindowParams {
            window_shape, strides, ..
        }) = op_params
        else {
            return Err(HoduError::VjpFunctionNotFound(
                "WindowingOp requires ReduceWindowParams".to_string(),
            ));
        };

        let input = inputs[0];
        let input_tensor = tensor_from_id(input);
        let input_shape = input_tensor.shape();
        let dtype = input_tensor.dtype();

        match self {
            WindowingOp::ReduceWindowMax | WindowingOp::ReduceWindowMin => {
                // For max/min pooling, gradient flows only through positions that were max/min
                // Upsample both output and gradient by repeating each element by stride amount

                // Upsample output
                let output_tensor = tensor_from_id(_output);
                let mut upsampled_output = output_tensor.clone();

                // Upsample gradient
                let grad_tensor = tensor_from_id(grad_output);
                let mut upsampled_grad = grad_tensor.clone();

                // For each dimension, if stride > 1, insert a dimension after it and broadcast
                for (dim_idx, stride) in strides.iter().enumerate().rev() {
                    let stride = *stride;
                    if stride > 1 {
                        // Upsample output
                        let current_shape = upsampled_output.shape();
                        let current_dims = current_shape.dims();

                        let mut expanded_shape = current_dims.to_vec();
                        expanded_shape.insert(dim_idx + 1, 1);
                        upsampled_output = upsampled_output.reshape(&expanded_shape)?;

                        let mut broadcast_shape = expanded_shape.clone();
                        broadcast_shape[dim_idx + 1] = stride;
                        upsampled_output = upsampled_output.broadcast(&broadcast_shape)?;

                        let mut final_shape = current_dims.to_vec();
                        final_shape[dim_idx] *= stride;
                        upsampled_output = upsampled_output.reshape(&final_shape)?;

                        // Upsample gradient (same process)
                        upsampled_grad = upsampled_grad.reshape(&expanded_shape)?;
                        upsampled_grad = upsampled_grad.broadcast(&broadcast_shape)?;
                        upsampled_grad = upsampled_grad.reshape(&final_shape)?;
                    }
                }

                // Broadcast to exact input shape (handles padding differences)
                upsampled_output = upsampled_output.broadcast(&input_shape)?;
                upsampled_grad = upsampled_grad.broadcast(&input_shape)?;

                // Create mask: input == upsampled_output
                let mask = input_tensor.eq(&upsampled_output)?;
                let mask_f = mask.to_dtype(dtype)?;

                // Multiply gradient by mask
                let result = upsampled_grad.mul(&mask_f)?;

                Ok(vec![result.id()])
            },
            WindowingOp::ReduceWindowMean => {
                // Mean: divide by window size and upsample gradient
                let grad_tensor = tensor_from_id(grad_output);
                let mut upsampled_grad = grad_tensor.clone();

                // Upsample gradient same way as Max/Min
                for (dim_idx, stride) in strides.iter().enumerate().rev() {
                    let stride = *stride;
                    if stride > 1 {
                        let current_shape = upsampled_grad.shape();
                        let current_dims = current_shape.dims();

                        let mut expanded_shape = current_dims.to_vec();
                        expanded_shape.insert(dim_idx + 1, 1);
                        upsampled_grad = upsampled_grad.reshape(&expanded_shape)?;

                        let mut broadcast_shape = expanded_shape.clone();
                        broadcast_shape[dim_idx + 1] = stride;
                        upsampled_grad = upsampled_grad.broadcast(&broadcast_shape)?;

                        let mut final_shape = current_dims.to_vec();
                        final_shape[dim_idx] *= stride;
                        upsampled_grad = upsampled_grad.reshape(&final_shape)?;
                    }
                }

                upsampled_grad = upsampled_grad.broadcast(&input_shape)?;

                let window_size: usize = window_shape.iter().product();
                let scale = Scalar::from_f32(1.0 / window_size as f32, dtype);
                let scaled_grad = upsampled_grad.mul_scalar(scale)?;
                Ok(vec![scaled_grad.id()])
            },
            WindowingOp::ReduceWindowSum => {
                // Sum: just upsample gradient
                let grad_tensor = tensor_from_id(grad_output);
                let mut upsampled_grad = grad_tensor.clone();

                // Upsample gradient same way as Max/Min
                for (dim_idx, stride) in strides.iter().enumerate().rev() {
                    let stride = *stride;
                    if stride > 1 {
                        let current_shape = upsampled_grad.shape();
                        let current_dims = current_shape.dims();

                        let mut expanded_shape = current_dims.to_vec();
                        expanded_shape.insert(dim_idx + 1, 1);
                        upsampled_grad = upsampled_grad.reshape(&expanded_shape)?;

                        let mut broadcast_shape = expanded_shape.clone();
                        broadcast_shape[dim_idx + 1] = stride;
                        upsampled_grad = upsampled_grad.broadcast(&broadcast_shape)?;

                        let mut final_shape = current_dims.to_vec();
                        final_shape[dim_idx] *= stride;
                        upsampled_grad = upsampled_grad.reshape(&final_shape)?;
                    }
                }

                upsampled_grad = upsampled_grad.broadcast(&input_shape)?;
                Ok(vec![upsampled_grad.id()])
            },
        }
    }
}
