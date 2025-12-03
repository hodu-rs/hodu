use super::VjpCompute;
use crate::{
    error::{HoduError, HoduResult},
    ops::{ConvOp, OpParams},
    tensor::{tensor_from_id, TensorId},
};

impl VjpCompute for ConvOp {
    fn compute_vjp(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        op_params: &OpParams,
    ) -> HoduResult<Vec<TensorId>> {
        match self {
            ConvOp::Conv1d => {
                // inputs: [input, weight]
                // Conv1d: input [N, Ci, L], weight [Co, Ci, K]
                let OpParams::Conv1d(p) = op_params else {
                    return Err(HoduError::VjpFunctionNotFound(
                        "Conv1d requires Conv1dParams".to_string(),
                    ));
                };
                if inputs.len() != 2 {
                    return Err(HoduError::InternalError("Conv1d requires 2 inputs".to_string()));
                }

                let input = tensor_from_id(inputs[0]);
                let weight = tensor_from_id(inputs[1]);
                let grad_output_tensor = tensor_from_id(grad_output);

                let channels_output = p.channels_output;
                let channels_input = p.channels_input;
                let kernel_size = p.kernel_size;
                let padding = p.padding;
                let stride = p.stride;
                let dilation = p.dilation;

                // Gradient w.r.t. input: use transposed convolution
                // Conv1d weight: [Co, Ci, K], Conv_transpose1d (backend): w [Co_out, Ci_in, K]
                // Transpose needed: [Co, Ci, K] -> [Ci, Co, K]
                let weight_transposed = weight.transpose(0, 1)?;
                let grad_input =
                    grad_output_tensor.conv_transpose1d(&weight_transposed, stride, padding, 0, dilation)?;

                // Gradient w.r.t. weight: conv1d_grad_weight(input, grad_output, weight_shape)
                let weight_shape = vec![channels_output, channels_input, kernel_size];
                let grad_weight =
                    input.conv1d_grad_weight(&grad_output_tensor, &weight_shape, stride, padding, dilation)?;

                Ok(vec![grad_input.id(), grad_weight.id()])
            },
            ConvOp::Conv2d => {
                // inputs: [input, weight]
                // Conv2d: input [N, Ci, H, W], weight [Co, Ci, Kh, Kw]
                let OpParams::Conv2d(p) = op_params else {
                    return Err(HoduError::VjpFunctionNotFound(
                        "Conv2d requires Conv2dParams".to_string(),
                    ));
                };
                if inputs.len() != 2 {
                    return Err(HoduError::InternalError("Conv2d requires 2 inputs".to_string()));
                }

                let input = tensor_from_id(inputs[0]);
                let weight = tensor_from_id(inputs[1]);
                let grad_output_tensor = tensor_from_id(grad_output);

                let kernel_height = p.kernel_height;
                let kernel_width = p.kernel_width;
                let channels_output = p.channels_output;
                let channels_input = p.channels_input;
                let padding = p.padding;
                let stride = p.stride;
                let dilation = p.dilation;

                // Gradient w.r.t. input: use transposed convolution
                // Conv2d: x [N, Ci, H, W] * w [Co, Ci, K, K] -> y [N, Co, H', W']
                // Backward: grad_y [N, Co, H', W'] -> grad_x [N, Ci, H, W]
                // Conv_transpose2d (backend): input [N, Ci_in, H, W] * w [Co_out, Ci_in, K, K] -> output [N, Co_out, H', W']
                // For backward: grad_y [N, Co, H', W'] needs w [Ci, Co, K, K] (transpose needed!)
                let weight_transposed = weight.transpose(0, 1)?; // [Co, Ci, K, K] -> [Ci, Co, K, K]
                let grad_input =
                    grad_output_tensor.conv_transpose2d(&weight_transposed, stride, padding, 0, dilation)?;

                // Gradient w.r.t. weight: conv2d_grad_weight(input, grad_output, weight_shape)
                let weight_shape_vec = vec![channels_output, channels_input, kernel_height, kernel_width];
                let grad_weight =
                    input.conv2d_grad_weight(&grad_output_tensor, &weight_shape_vec, stride, padding, dilation)?;

                Ok(vec![grad_input.id(), grad_weight.id()])
            },
            ConvOp::Conv3d => {
                // inputs: [input, weight]
                // Conv3d: input [N, Ci, D, H, W], weight [Co, Ci, Kd, Kh, Kw]
                let OpParams::Conv3d(p) = op_params else {
                    return Err(HoduError::VjpFunctionNotFound(
                        "Conv3d requires Conv3dParams".to_string(),
                    ));
                };
                if inputs.len() != 2 {
                    return Err(HoduError::InternalError("Conv3d requires 2 inputs".to_string()));
                }

                let input = tensor_from_id(inputs[0]);
                let weight = tensor_from_id(inputs[1]);
                let grad_output_tensor = tensor_from_id(grad_output);

                let kernel_depth = p.kernel_depth;
                let kernel_height = p.kernel_height;
                let kernel_width = p.kernel_width;
                let channels_output = p.channels_output;
                let channels_input = p.channels_input;
                let padding = p.padding;
                let stride = p.stride;
                let dilation = p.dilation;

                // Gradient w.r.t. input: use transposed convolution
                // Conv3d weight: [Co, Ci, Kd, Kh, Kw], Conv_transpose3d (backend): w [Co_out, Ci_in, ...]
                // Transpose needed: [Co, Ci, ...] -> [Ci, Co, ...]
                let weight_transposed = weight.transpose(0, 1)?;
                let grad_input =
                    grad_output_tensor.conv_transpose3d(&weight_transposed, stride, padding, 0, dilation)?;

                // Gradient w.r.t. weight: conv3d_grad_weight(input, grad_output, weight_shape)
                let weight_shape = vec![
                    channels_output,
                    channels_input,
                    kernel_depth,
                    kernel_height,
                    kernel_width,
                ];
                let grad_weight =
                    input.conv3d_grad_weight(&grad_output_tensor, &weight_shape, stride, padding, dilation)?;

                Ok(vec![grad_input.id(), grad_weight.id()])
            },
            ConvOp::ConvTranspose1d => {
                // inputs: [input, weight]
                // ConvTranspose1d: input [N, Ci, L_in], weight [Ci, Co, K]
                let OpParams::ConvTranspose1d(p) = op_params else {
                    return Err(HoduError::VjpFunctionNotFound(
                        "ConvTranspose1d requires ConvTranspose1dParams".to_string(),
                    ));
                };
                if inputs.len() != 2 {
                    return Err(HoduError::InternalError(
                        "ConvTranspose1d requires 2 inputs".to_string(),
                    ));
                }

                let input = tensor_from_id(inputs[0]);
                let weight = tensor_from_id(inputs[1]);
                let grad_output_tensor = tensor_from_id(grad_output);

                let channels_output = p.channels_output;
                let channels_input = p.channels_input;
                let kernel_size = p.kernel_size;
                let padding = p.padding;
                let stride = p.stride;
                let dilation = p.dilation;

                // Gradient w.r.t. input: Use regular Conv1d
                // weight shape for conv1d is [Co, Ci, K], but we need to transpose channels
                let grad_input = grad_output_tensor.conv1d(&weight, stride, padding, dilation)?;

                // Gradient w.r.t. weight: conv_transpose1d_grad_weight(input, grad_output, weight_shape)
                let weight_shape = vec![channels_input, channels_output, kernel_size];
                let grad_weight = input.conv_transpose1d_grad_weight(
                    &grad_output_tensor,
                    &weight_shape,
                    stride,
                    padding,
                    dilation,
                )?;

                Ok(vec![grad_input.id(), grad_weight.id()])
            },
            ConvOp::ConvTranspose2d => {
                // inputs: [input, weight]
                // ConvTranspose2d: input [N, Ci, H_in, W_in], weight [Ci, Co, Kh, Kw]
                let OpParams::ConvTranspose2d(p) = op_params else {
                    return Err(HoduError::VjpFunctionNotFound(
                        "ConvTranspose2d requires ConvTranspose2dParams".to_string(),
                    ));
                };
                if inputs.len() != 2 {
                    return Err(HoduError::InternalError(
                        "ConvTranspose2d requires 2 inputs".to_string(),
                    ));
                }

                let input = tensor_from_id(inputs[0]);
                let weight = tensor_from_id(inputs[1]);
                let grad_output_tensor = tensor_from_id(grad_output);

                let kernel_height = p.kernel_height;
                let kernel_width = p.kernel_width;
                let channels_output = p.channels_output;
                let channels_input = p.channels_input;
                let padding = p.padding;
                let stride = p.stride;
                let dilation = p.dilation;

                // Gradient w.r.t. input: Use regular Conv2d
                let grad_input = grad_output_tensor.conv2d(&weight, stride, padding, dilation)?;

                // Gradient w.r.t. weight: conv_transpose2d_grad_weight
                let weight_shape = vec![channels_input, channels_output, kernel_height, kernel_width];
                let grad_weight = input.conv_transpose2d_grad_weight(
                    &grad_output_tensor,
                    &weight_shape,
                    stride,
                    padding,
                    dilation,
                )?;

                Ok(vec![grad_input.id(), grad_weight.id()])
            },
            ConvOp::ConvTranspose3d => {
                // inputs: [input, weight]
                // ConvTranspose3d: input [N, Ci, D_in, H_in, W_in], weight [Ci, Co, Kd, Kh, Kw]
                let OpParams::ConvTranspose3d(p) = op_params else {
                    return Err(HoduError::VjpFunctionNotFound(
                        "ConvTranspose3d requires ConvTranspose3dParams".to_string(),
                    ));
                };
                if inputs.len() != 2 {
                    return Err(HoduError::InternalError(
                        "ConvTranspose3d requires 2 inputs".to_string(),
                    ));
                }

                let input = tensor_from_id(inputs[0]);
                let weight = tensor_from_id(inputs[1]);
                let grad_output_tensor = tensor_from_id(grad_output);

                let kernel_depth = p.kernel_depth;
                let kernel_height = p.kernel_height;
                let kernel_width = p.kernel_width;
                let channels_output = p.channels_output;
                let channels_input = p.channels_input;
                let padding = p.padding;
                let stride = p.stride;
                let dilation = p.dilation;

                // Gradient w.r.t. input: Use regular Conv3d
                let grad_input = grad_output_tensor.conv3d(&weight, stride, padding, dilation)?;

                // Gradient w.r.t. weight: conv_transpose3d_grad_weight
                let weight_shape = vec![
                    channels_input,
                    channels_output,
                    kernel_depth,
                    kernel_height,
                    kernel_width,
                ];
                let grad_weight = input.conv_transpose3d_grad_weight(
                    &grad_output_tensor,
                    &weight_shape,
                    stride,
                    padding,
                    dilation,
                )?;

                Ok(vec![grad_input.id(), grad_weight.id()])
            },
            // GradWeight operations don't need gradients (gradient of gradient not needed)
            ConvOp::Conv1dGradWeight
            | ConvOp::Conv2dGradWeight
            | ConvOp::Conv3dGradWeight
            | ConvOp::ConvTranspose1dGradWeight
            | ConvOp::ConvTranspose2dGradWeight
            | ConvOp::ConvTranspose3dGradWeight => Err(HoduError::InternalError(
                "Gradient of gradient weight operation not supported".to_string(),
            )),
        }
    }
}
