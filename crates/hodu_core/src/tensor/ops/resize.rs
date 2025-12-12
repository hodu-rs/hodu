use crate::{
    error::{HoduError, HoduResult},
    op_params::{OpParams, ResizeCoordTransform, ResizeMode, ResizeNearestMode, ResizeParams},
    ops::{Op, ResizeOp},
    tensor::{create_builder_tensor, from_storage_with_context, gradient, Tensor},
    types::{Layout, Shape},
    utils::valid::{validate_dtype_for_device, validate_dtype_for_op, validate_requires_grad_for_op},
};

impl Tensor {
    /// Resize the spatial dimensions of a tensor using interpolation.
    ///
    /// This operation resizes the height and width (and optionally depth) dimensions
    /// of a tensor while preserving batch and channel dimensions. The tensor must be
    /// in NCHW format (4D) or NCDHW format (5D).
    ///
    /// # Arguments
    /// * `output_size` - The target spatial dimensions [H, W] for 4D or [D, H, W] for 5D tensors
    /// * `mode` - Interpolation mode: "nearest", "bilinear"/"linear", or "bicubic"/"cubic"
    /// * `coord_transform` - Coordinate transformation: "half_pixel", "asymmetric", "align_corners", or "pytorch_half_pixel"
    /// * `nearest_mode` - Rounding for nearest interpolation: "floor", "ceil", "round_prefer_floor", or "round_prefer_ceil"
    ///
    /// # Returns
    /// A new tensor with the resized spatial dimensions
    ///
    /// # Example
    /// ```ignore
    /// // Resize 2x2 image to 4x4 using bilinear interpolation
    /// let input = Tensor::randn(&[1, 3, 2, 2])?;
    /// let output = input.resize(&[4, 4], "bilinear", "half_pixel", "floor")?;
    /// // output shape: [1, 3, 4, 4]
    /// ```
    pub fn resize(
        &self,
        output_size: &[usize],
        mode: &str,
        coord_transform: &str,
        nearest_mode: &str,
    ) -> HoduResult<Self> {
        let input_shape = self.shape();
        let rank = input_shape.ndim();

        // Validate rank (must be 4D NCHW or 5D NCDHW)
        if rank != 4 && rank != 5 {
            return Err(HoduError::InvalidLayout {
                reason: format!("resize requires 4D (NCHW) or 5D (NCDHW) tensor, got {}D", rank),
            });
        }

        let spatial_dims = rank - 2; // 2 for 4D, 3 for 5D
        if output_size.len() != spatial_dims {
            return Err(HoduError::InvalidLayout {
                reason: format!(
                    "output_size length {} must match spatial dimensions {}",
                    output_size.len(),
                    spatial_dims
                ),
            });
        }

        let resize_mode = match mode.to_lowercase().as_str() {
            "nearest" => ResizeMode::Nearest,
            "linear" | "bilinear" | "trilinear" => ResizeMode::Linear,
            "cubic" | "bicubic" => ResizeMode::Cubic,
            _ => {
                return Err(HoduError::InvalidLayout {
                    reason: format!(
                        "invalid resize mode '{}'. Must be one of: 'nearest', 'bilinear', 'linear', 'bicubic', 'cubic'",
                        mode
                    ),
                })
            },
        };

        let resize_coord_transform = match coord_transform.to_lowercase().as_str() {
            "half_pixel" => ResizeCoordTransform::HalfPixel,
            "asymmetric" => ResizeCoordTransform::Asymmetric,
            "align_corners" => ResizeCoordTransform::AlignCorners,
            "pytorch_half_pixel" => ResizeCoordTransform::PytorchHalfPixel,
            _ => {
                return Err(HoduError::InvalidLayout {
                    reason: format!(
                        "invalid coord_transform '{}'. Must be one of: 'half_pixel', 'asymmetric', 'align_corners', 'pytorch_half_pixel'",
                        coord_transform
                    ),
                })
            },
        };

        let resize_nearest_mode = match nearest_mode.to_lowercase().as_str() {
            "floor" => ResizeNearestMode::Floor,
            "ceil" => ResizeNearestMode::Ceil,
            "round_prefer_floor" => ResizeNearestMode::RoundPreferFloor,
            "round_prefer_ceil" => ResizeNearestMode::RoundPreferCeil,
            _ => {
                return Err(HoduError::InvalidLayout {
                    reason: format!(
                        "invalid nearest_mode '{}'. Must be one of: 'floor', 'ceil', 'round_prefer_floor', 'round_prefer_ceil'",
                        nearest_mode
                    ),
                })
            },
        };

        validate_dtype_for_device(self.dtype(), self.device())?;
        validate_dtype_for_op(self.dtype(), Op::Resize(ResizeOp::Resize))?;
        let validate_requires_grad = validate_requires_grad_for_op(Op::Resize(ResizeOp::Resize));

        // Build output shape: [N, C, output_size...]
        let input_dims = input_shape.dims();
        let mut output_dims = Vec::with_capacity(rank);
        output_dims.push(input_dims[0]); // batch
        output_dims.push(input_dims[1]); // channels
        output_dims.extend_from_slice(output_size); // spatial dims

        let op_params = OpParams::Resize(ResizeParams {
            output_size: output_size.to_vec(),
            mode: resize_mode,
            coord_transform: resize_coord_transform,
            nearest_mode: resize_nearest_mode,
        });

        let input_layout = self.layout();
        let result_layout = Layout::from_shape(&Shape::from(output_dims.clone()));
        let requires_grad = self.is_requires_grad() && validate_requires_grad;

        if crate::snapshot::capture::is_active() {
            let (result_id, result_tensor) = create_builder_tensor(result_layout.clone(), self.dtype(), requires_grad);

            crate::snapshot::capture::capture_operation(
                Op::Resize(ResizeOp::Resize),
                Some(op_params.clone()),
                vec![self.id()],
                result_id,
                vec![input_layout],
                result_layout,
            )?;

            if requires_grad {
                gradient::record_operation(vec![self.id()], result_id, Op::Resize(ResizeOp::Resize), op_params)?;
            }

            Ok(result_tensor)
        } else {
            let storage = self.with_storage(|input_storage| {
                input_storage.call_ops_resize(
                    &self.layout(),
                    &output_dims,
                    resize_mode,
                    resize_coord_transform,
                    resize_nearest_mode,
                )
            })?;

            let result = from_storage_with_context(storage, result_layout, true, requires_grad);

            if !gradient::is_computing_gradients() && requires_grad {
                gradient::record_operation(vec![self.id()], result.id(), Op::Resize(ResizeOp::Resize), op_params)?;
            }

            Ok(result)
        }
    }

    /// Resize using nearest neighbor interpolation
    pub fn resize_nearest(&self, output_size: &[usize]) -> HoduResult<Self> {
        self.resize(output_size, "nearest", "asymmetric", "floor")
    }

    /// Resize using bilinear interpolation with half_pixel coordinate transform
    pub fn resize_bilinear(&self, output_size: &[usize]) -> HoduResult<Self> {
        self.resize(output_size, "bilinear", "half_pixel", "floor")
    }

    /// Resize using bilinear interpolation with align_corners coordinate transform
    pub fn resize_bilinear_align_corners(&self, output_size: &[usize]) -> HoduResult<Self> {
        self.resize(output_size, "bilinear", "align_corners", "floor")
    }

    /// Resize using bicubic interpolation with half_pixel coordinate transform
    pub fn resize_bicubic(&self, output_size: &[usize]) -> HoduResult<Self> {
        self.resize(output_size, "bicubic", "half_pixel", "floor")
    }
}
