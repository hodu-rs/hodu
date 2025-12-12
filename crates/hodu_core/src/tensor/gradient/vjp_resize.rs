use super::VjpCompute;
use crate::{
    error::{HoduError, HoduResult},
    op_params::{OpParams, ResizeMode, ResizeParams},
    ops::ResizeOp,
    tensor::{tensor_from_id, TensorId},
};

impl VjpCompute for ResizeOp {
    fn compute_vjp(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        op_params: &OpParams,
    ) -> HoduResult<Vec<TensorId>> {
        let OpParams::Resize(ResizeParams {
            mode,
            coord_transform,
            nearest_mode,
            ..
        }) = op_params
        else {
            return Err(HoduError::VjpFunctionNotFound(
                "ResizeOp requires ResizeParams".to_string(),
            ));
        };

        match self {
            ResizeOp::Resize => {
                let input = inputs[0];
                let input_tensor = tensor_from_id(input);
                let input_shape = input_tensor.shape();
                let input_dims = input_shape.dims();

                let grad_tensor = tensor_from_id(grad_output);

                // For resize, the gradient flows back by resizing grad_output to input shape
                // This is valid for bilinear/bicubic interpolation
                // For nearest neighbor, there's no well-defined gradient (we use bilinear for backward)

                // Get spatial dimensions from input (skip batch and channel)
                let spatial_dims: Vec<usize> = input_dims[2..].to_vec();

                // For nearest mode, there's technically no gradient, but we can approximate
                // by using bilinear interpolation for the backward pass
                let backward_mode = match mode {
                    ResizeMode::Nearest => "bilinear", // Use bilinear for backward pass
                    ResizeMode::Linear => "bilinear",
                    ResizeMode::Cubic => "bicubic",
                };

                let coord_transform_str = match coord_transform {
                    crate::op_params::ResizeCoordTransform::HalfPixel => "half_pixel",
                    crate::op_params::ResizeCoordTransform::Asymmetric => "asymmetric",
                    crate::op_params::ResizeCoordTransform::AlignCorners => "align_corners",
                    crate::op_params::ResizeCoordTransform::PytorchHalfPixel => "pytorch_half_pixel",
                };

                let nearest_mode_str = match nearest_mode {
                    crate::op_params::ResizeNearestMode::Floor => "floor",
                    crate::op_params::ResizeNearestMode::Ceil => "ceil",
                    crate::op_params::ResizeNearestMode::RoundPreferFloor => "round_prefer_floor",
                    crate::op_params::ResizeNearestMode::RoundPreferCeil => "round_prefer_ceil",
                };

                // Resize grad_output back to input spatial dimensions
                let grad_input =
                    grad_tensor.resize(&spatial_dims, backward_mode, coord_transform_str, nearest_mode_str)?;

                Ok(vec![grad_input.id()])
            },
        }
    }
}
