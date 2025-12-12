use crate::module::Module;
use hodu_core::{error::HoduResult, tensor::Tensor};

#[derive(Module, Clone)]
pub struct AdaptiveAvgPool1D {
    output_size: usize,
}

impl AdaptiveAvgPool1D {
    pub fn new(output_size: usize) -> Self {
        Self { output_size }
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // AdaptiveAvgPool1D: input [N, C, L]
        let input_shape = input.shape();
        let rank = input_shape.ndim();

        if rank != 3 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "AdaptiveAvgPool1D expects 3D input [N, C, L], got {}D",
                rank
            )));
        }

        let input_length = input_shape[2];

        let stride = (input_length as f64 / self.output_size as f64).floor() as usize;
        let kernel_size = input_length - (self.output_size - 1) * stride;

        let padding = vec![(0, 0), (0, 0), (0, 0)];
        let window_shape = vec![1, 1, kernel_size];
        let strides = vec![1, 1, stride];

        input.reduce_window(&window_shape, &strides, &padding, "mean")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

#[derive(Module, Clone)]
pub struct AdaptiveAvgPool2D {
    output_size: (usize, usize),
}

impl AdaptiveAvgPool2D {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // AdaptiveAvgPool2D: input [N, C, H, W]
        let input_shape = input.shape();
        let rank = input_shape.ndim();

        if rank != 4 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "AdaptiveAvgPool2D expects 4D input [N, C, H, W], got {}D",
                rank
            )));
        }

        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let stride_h = (input_height as f64 / self.output_size.0 as f64).floor() as usize;
        let stride_w = (input_width as f64 / self.output_size.1 as f64).floor() as usize;
        let kernel_h = input_height - (self.output_size.0 - 1) * stride_h;
        let kernel_w = input_width - (self.output_size.1 - 1) * stride_w;

        let padding = vec![(0, 0), (0, 0), (0, 0), (0, 0)];
        let window_shape = vec![1, 1, kernel_h, kernel_w];
        let strides = vec![1, 1, stride_h, stride_w];

        input.reduce_window(&window_shape, &strides, &padding, "mean")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

#[derive(Module, Clone)]
pub struct AdaptiveAvgPool3D {
    output_size: (usize, usize, usize),
}

impl AdaptiveAvgPool3D {
    pub fn new(output_size: (usize, usize, usize)) -> Self {
        Self { output_size }
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // AdaptiveAvgPool3D: input [N, C, D, H, W]
        let input_shape = input.shape();
        let rank = input_shape.ndim();

        if rank != 5 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "AdaptiveAvgPool3D expects 5D input [N, C, D, H, W], got {}D",
                rank
            )));
        }

        let input_depth = input_shape[2];
        let input_height = input_shape[3];
        let input_width = input_shape[4];

        let stride_d = (input_depth as f64 / self.output_size.0 as f64).floor() as usize;
        let stride_h = (input_height as f64 / self.output_size.1 as f64).floor() as usize;
        let stride_w = (input_width as f64 / self.output_size.2 as f64).floor() as usize;
        let kernel_d = input_depth - (self.output_size.0 - 1) * stride_d;
        let kernel_h = input_height - (self.output_size.1 - 1) * stride_h;
        let kernel_w = input_width - (self.output_size.2 - 1) * stride_w;

        let padding = vec![(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)];
        let window_shape = vec![1, 1, kernel_d, kernel_h, kernel_w];
        let strides = vec![1, 1, stride_d, stride_h, stride_w];

        input.reduce_window(&window_shape, &strides, &padding, "mean")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

#[derive(Module, Clone)]
pub struct AdaptiveMaxPool1D {
    output_size: usize,
}

impl AdaptiveMaxPool1D {
    pub fn new(output_size: usize) -> Self {
        Self { output_size }
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // AdaptiveMaxPool1D: input [N, C, L]
        let input_shape = input.shape();
        let rank = input_shape.ndim();

        if rank != 3 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "AdaptiveMaxPool1D expects 3D input [N, C, L], got {}D",
                rank
            )));
        }

        let input_length = input_shape[2];

        // Calculate kernel_size and stride for adaptive pooling
        let stride = (input_length as f64 / self.output_size as f64).floor() as usize;
        let kernel_size = input_length - (self.output_size - 1) * stride;

        let padding = vec![(0, 0), (0, 0), (0, 0)];
        let window_shape = vec![1, 1, kernel_size];
        let strides = vec![1, 1, stride];

        input.reduce_window(&window_shape, &strides, &padding, "max")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

#[derive(Module, Clone)]
pub struct AdaptiveMaxPool2D {
    output_size: (usize, usize),
}

impl AdaptiveMaxPool2D {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // AdaptiveMaxPool2D: input [N, C, H, W]
        let input_shape = input.shape();
        let rank = input_shape.ndim();

        if rank != 4 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "AdaptiveMaxPool2D expects 4D input [N, C, H, W], got {}D",
                rank
            )));
        }

        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let stride_h = (input_height as f64 / self.output_size.0 as f64).floor() as usize;
        let stride_w = (input_width as f64 / self.output_size.1 as f64).floor() as usize;
        let kernel_h = input_height - (self.output_size.0 - 1) * stride_h;
        let kernel_w = input_width - (self.output_size.1 - 1) * stride_w;

        let padding = vec![(0, 0), (0, 0), (0, 0), (0, 0)];
        let window_shape = vec![1, 1, kernel_h, kernel_w];
        let strides = vec![1, 1, stride_h, stride_w];

        input.reduce_window(&window_shape, &strides, &padding, "max")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

#[derive(Module, Clone)]
pub struct AdaptiveMaxPool3D {
    output_size: (usize, usize, usize),
}

impl AdaptiveMaxPool3D {
    pub fn new(output_size: (usize, usize, usize)) -> Self {
        Self { output_size }
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // AdaptiveMaxPool3D: input [N, C, D, H, W]
        let input_shape = input.shape();
        let rank = input_shape.ndim();

        if rank != 5 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "AdaptiveMaxPool3D expects 5D input [N, C, D, H, W], got {}D",
                rank
            )));
        }

        let input_depth = input_shape[2];
        let input_height = input_shape[3];
        let input_width = input_shape[4];

        let stride_d = (input_depth as f64 / self.output_size.0 as f64).floor() as usize;
        let stride_h = (input_height as f64 / self.output_size.1 as f64).floor() as usize;
        let stride_w = (input_width as f64 / self.output_size.2 as f64).floor() as usize;
        let kernel_d = input_depth - (self.output_size.0 - 1) * stride_d;
        let kernel_h = input_height - (self.output_size.1 - 1) * stride_h;
        let kernel_w = input_width - (self.output_size.2 - 1) * stride_w;

        let padding = vec![(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)];
        let window_shape = vec![1, 1, kernel_d, kernel_h, kernel_w];
        let strides = vec![1, 1, stride_d, stride_h, stride_w];

        input.reduce_window(&window_shape, &strides, &padding, "max")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

#[derive(Module, Clone)]
pub struct AvgPool1D {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl AvgPool1D {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // AvgPool1D: input [N, C, L]
        let input_shape = input.shape();
        let rank = input_shape.ndim();

        if rank != 3 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "AvgPool1D expects 3D input [N, C, L], got {}D",
                rank
            )));
        }

        let padding = vec![(0, 0), (0, 0), (self.padding, self.padding)];
        let window_shape = vec![1, 1, self.kernel_size];
        let strides = vec![1, 1, self.stride];

        input.reduce_window(&window_shape, &strides, &padding, "mean")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

#[derive(Module, Clone)]
pub struct AvgPool2D {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl AvgPool2D {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // AvgPool2D: input [N, C, H, W]
        let input_shape = input.shape();
        let rank = input_shape.ndim();

        if rank != 4 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "AvgPool2D expects 4D input [N, C, H, W], got {}D",
                rank
            )));
        }

        let padding = vec![
            (0, 0),
            (0, 0),
            (self.padding, self.padding),
            (self.padding, self.padding),
        ];
        let window_shape = vec![1, 1, self.kernel_size, self.kernel_size];
        let strides = vec![1, 1, self.stride, self.stride];

        input.reduce_window(&window_shape, &strides, &padding, "mean")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

#[derive(Module, Clone)]
pub struct AvgPool3D {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl AvgPool3D {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // AvgPool3D: input [N, C, D, H, W]
        let input_shape = input.shape();
        let rank = input_shape.ndim();

        if rank != 5 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "AvgPool3D expects 5D input [N, C, D, H, W], got {}D",
                rank
            )));
        }

        let padding = vec![
            (0, 0),
            (0, 0),
            (self.padding, self.padding),
            (self.padding, self.padding),
            (self.padding, self.padding),
        ];
        let window_shape = vec![1, 1, self.kernel_size, self.kernel_size, self.kernel_size];
        let strides = vec![1, 1, self.stride, self.stride, self.stride];

        input.reduce_window(&window_shape, &strides, &padding, "mean")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

#[derive(Module, Clone)]
pub struct MaxPool1D {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl MaxPool1D {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // MaxPool1D: input [N, C, L]
        let input_shape = input.shape();
        let rank = input_shape.ndim();

        if rank != 3 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "MaxPool1D expects 3D input [N, C, L], got {}D",
                rank
            )));
        }

        // Build padding, window_shape, and strides for all dimensions
        let padding = vec![(0, 0), (0, 0), (self.padding, self.padding)];
        let window_shape = vec![1, 1, self.kernel_size];
        let strides = vec![1, 1, self.stride];

        input.reduce_window(&window_shape, &strides, &padding, "max")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

#[derive(Module, Clone)]
pub struct MaxPool2D {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl MaxPool2D {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // MaxPool2D: input [N, C, H, W]
        let input_shape = input.shape();
        let rank = input_shape.ndim();

        if rank != 4 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "MaxPool2D expects 4D input [N, C, H, W], got {}D",
                rank
            )));
        }

        let padding = vec![
            (0, 0),
            (0, 0),
            (self.padding, self.padding),
            (self.padding, self.padding),
        ];
        let window_shape = vec![1, 1, self.kernel_size, self.kernel_size];
        let strides = vec![1, 1, self.stride, self.stride];

        input.reduce_window(&window_shape, &strides, &padding, "max")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

#[derive(Module, Clone)]
pub struct MaxPool3D {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl MaxPool3D {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // MaxPool3D: input [N, C, D, H, W]
        let input_shape = input.shape();
        let rank = input_shape.ndim();

        if rank != 5 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "MaxPool3D expects 5D input [N, C, D, H, W], got {}D",
                rank
            )));
        }

        let padding = vec![
            (0, 0),
            (0, 0),
            (self.padding, self.padding),
            (self.padding, self.padding),
            (self.padding, self.padding),
        ];
        let window_shape = vec![1, 1, self.kernel_size, self.kernel_size, self.kernel_size];
        let strides = vec![1, 1, self.stride, self.stride, self.stride];

        input.reduce_window(&window_shape, &strides, &padding, "max")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

// ============================================================================
// GlobalAvgPool
// ============================================================================

/// Global Average Pooling for 1D inputs.
///
/// Pools over the entire spatial dimension to produce [N, C, 1].
#[derive(Module, Clone)]
pub struct GlobalAvgPool1D;

impl GlobalAvgPool1D {
    pub fn new() -> Self {
        Self
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.ndim() != 3 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "GlobalAvgPool1D expects 3D input [N, C, L], got {}D",
                input_shape.ndim()
            )));
        }
        // Mean over L dimension
        input.mean(&[2], true)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

impl Default for GlobalAvgPool1D {
    fn default() -> Self {
        Self::new()
    }
}

/// Global Average Pooling for 2D inputs.
///
/// Pools over the entire spatial dimensions to produce [N, C, 1, 1].
#[derive(Module, Clone)]
pub struct GlobalAvgPool2D;

impl GlobalAvgPool2D {
    pub fn new() -> Self {
        Self
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.ndim() != 4 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "GlobalAvgPool2D expects 4D input [N, C, H, W], got {}D",
                input_shape.ndim()
            )));
        }
        // Mean over H, W dimensions
        input.mean(&[2, 3], true)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

impl Default for GlobalAvgPool2D {
    fn default() -> Self {
        Self::new()
    }
}

/// Global Average Pooling for 3D inputs.
///
/// Pools over the entire spatial dimensions to produce [N, C, 1, 1, 1].
#[derive(Module, Clone)]
pub struct GlobalAvgPool3D;

impl GlobalAvgPool3D {
    pub fn new() -> Self {
        Self
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.ndim() != 5 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "GlobalAvgPool3D expects 5D input [N, C, D, H, W], got {}D",
                input_shape.ndim()
            )));
        }
        // Mean over D, H, W dimensions
        input.mean(&[2, 3, 4], true)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

impl Default for GlobalAvgPool3D {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// GlobalMaxPool
// ============================================================================

/// Global Max Pooling for 1D inputs.
///
/// Pools over the entire spatial dimension to produce [N, C, 1].
#[derive(Module, Clone)]
pub struct GlobalMaxPool1D;

impl GlobalMaxPool1D {
    pub fn new() -> Self {
        Self
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.ndim() != 3 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "GlobalMaxPool1D expects 3D input [N, C, L], got {}D",
                input_shape.ndim()
            )));
        }
        // Max over L dimension
        input.max(&[2], true)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

impl Default for GlobalMaxPool1D {
    fn default() -> Self {
        Self::new()
    }
}

/// Global Max Pooling for 2D inputs.
///
/// Pools over the entire spatial dimensions to produce [N, C, 1, 1].
#[derive(Module, Clone)]
pub struct GlobalMaxPool2D;

impl GlobalMaxPool2D {
    pub fn new() -> Self {
        Self
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.ndim() != 4 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "GlobalMaxPool2D expects 4D input [N, C, H, W], got {}D",
                input_shape.ndim()
            )));
        }
        // Max over H, W dimensions
        input.max(&[2, 3], true)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

impl Default for GlobalMaxPool2D {
    fn default() -> Self {
        Self::new()
    }
}

/// Global Max Pooling for 3D inputs.
///
/// Pools over the entire spatial dimensions to produce [N, C, 1, 1, 1].
#[derive(Module, Clone)]
pub struct GlobalMaxPool3D;

impl GlobalMaxPool3D {
    pub fn new() -> Self {
        Self
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.ndim() != 5 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "GlobalMaxPool3D expects 5D input [N, C, D, H, W], got {}D",
                input_shape.ndim()
            )));
        }
        // Max over D, H, W dimensions
        input.max(&[2, 3, 4], true)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

impl Default for GlobalMaxPool3D {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// FractionalMaxPool
// ============================================================================

/// Fractional Max Pooling for 2D inputs.
///
/// Applies max pooling with fractional output sizes, using random or
/// pseudo-random pooling regions.
///
/// Reference: "Fractional Max-Pooling" by Benjamin Graham (2014)
#[derive(Module, Clone)]
pub struct FractionalMaxPool2D {
    output_size: Option<(usize, usize)>,
    output_ratio: Option<(f64, f64)>,
}

impl FractionalMaxPool2D {
    /// Creates FractionalMaxPool2D with a target output size.
    pub fn with_output_size(output_size: (usize, usize)) -> Self {
        Self {
            output_size: Some(output_size),
            output_ratio: None,
        }
    }

    /// Creates FractionalMaxPool2D with a target output ratio.
    ///
    /// Output size will be input_size * output_ratio.
    pub fn with_output_ratio(output_ratio: (f64, f64)) -> Self {
        Self {
            output_size: None,
            output_ratio: Some(output_ratio),
        }
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.ndim() != 4 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "FractionalMaxPool2D expects 4D input [N, C, H, W], got {}D",
                input_shape.ndim()
            )));
        }

        let n = input_shape[0];
        let c = input_shape[1];
        let h = input_shape[2];
        let w = input_shape[3];

        let (out_h, out_w) = if let Some((oh, ow)) = self.output_size {
            (oh, ow)
        } else if let Some((rh, rw)) = self.output_ratio {
            ((h as f64 * rh) as usize, (w as f64 * rw) as usize)
        } else {
            return Err(hodu_core::error::HoduError::InternalError(
                "FractionalMaxPool2D requires output_size or output_ratio".to_string(),
            ));
        };

        // Generate fractional pooling sequence
        let h_sequence = generate_fractional_sequence(h, out_h);
        let w_sequence = generate_fractional_sequence(w, out_w);

        // Collect pooled results
        let mut results = Vec::with_capacity(n * c * out_h * out_w);

        for batch in 0..n {
            for channel in 0..c {
                for i in 0..out_h {
                    let h_start = h_sequence[i];
                    let h_end = h_sequence[i + 1];

                    for j in 0..out_w {
                        let w_start = w_sequence[j];
                        let w_end = w_sequence[j + 1];

                        // Extract region and compute max
                        let region = input
                            .slice(0, batch, Some(batch + 1), 1)?
                            .slice(1, channel, Some(channel + 1), 1)?
                            .slice(2, h_start, Some(h_end), 1)?
                            .slice(3, w_start, Some(w_end), 1)?;

                        let max_val = region.max(&[0, 1, 2, 3], false)?;
                        results.push(max_val);
                    }
                }
            }
        }

        // Stack and reshape results
        let result_refs: Vec<&Tensor> = results.iter().collect();
        let stacked = Tensor::stack(&result_refs, 0)?;
        stacked.reshape([n, c, out_h, out_w])
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

/// Fractional Max Pooling for 3D inputs.
#[derive(Module, Clone)]
pub struct FractionalMaxPool3D {
    output_size: Option<(usize, usize, usize)>,
    output_ratio: Option<(f64, f64, f64)>,
}

impl FractionalMaxPool3D {
    /// Creates FractionalMaxPool3D with a target output size.
    pub fn with_output_size(output_size: (usize, usize, usize)) -> Self {
        Self {
            output_size: Some(output_size),
            output_ratio: None,
        }
    }

    /// Creates FractionalMaxPool3D with a target output ratio.
    pub fn with_output_ratio(output_ratio: (f64, f64, f64)) -> Self {
        Self {
            output_size: None,
            output_ratio: Some(output_ratio),
        }
    }

    fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let input_shape = input.shape();
        if input_shape.ndim() != 5 {
            return Err(hodu_core::error::HoduError::InternalError(format!(
                "FractionalMaxPool3D expects 5D input [N, C, D, H, W], got {}D",
                input_shape.ndim()
            )));
        }

        let n = input_shape[0];
        let c = input_shape[1];
        let d = input_shape[2];
        let h = input_shape[3];
        let w = input_shape[4];

        let (out_d, out_h, out_w) = if let Some((od, oh, ow)) = self.output_size {
            (od, oh, ow)
        } else if let Some((rd, rh, rw)) = self.output_ratio {
            (
                (d as f64 * rd) as usize,
                (h as f64 * rh) as usize,
                (w as f64 * rw) as usize,
            )
        } else {
            return Err(hodu_core::error::HoduError::InternalError(
                "FractionalMaxPool3D requires output_size or output_ratio".to_string(),
            ));
        };

        // Generate fractional pooling sequences
        let d_sequence = generate_fractional_sequence(d, out_d);
        let h_sequence = generate_fractional_sequence(h, out_h);
        let w_sequence = generate_fractional_sequence(w, out_w);

        // Collect pooled results
        let mut results = Vec::with_capacity(n * c * out_d * out_h * out_w);

        for batch in 0..n {
            for channel in 0..c {
                for di in 0..out_d {
                    let d_start = d_sequence[di];
                    let d_end = d_sequence[di + 1];

                    for hi in 0..out_h {
                        let h_start = h_sequence[hi];
                        let h_end = h_sequence[hi + 1];

                        for wi in 0..out_w {
                            let w_start = w_sequence[wi];
                            let w_end = w_sequence[wi + 1];

                            let region = input
                                .slice(0, batch, Some(batch + 1), 1)?
                                .slice(1, channel, Some(channel + 1), 1)?
                                .slice(2, d_start, Some(d_end), 1)?
                                .slice(3, h_start, Some(h_end), 1)?
                                .slice(4, w_start, Some(w_end), 1)?;

                            let max_val = region.max(&[0, 1, 2, 3, 4], false)?;
                            results.push(max_val);
                        }
                    }
                }
            }
        }

        let result_refs: Vec<&Tensor> = results.iter().collect();
        let stacked = Tensor::stack(&result_refs, 0)?;
        stacked.reshape([n, c, out_d, out_h, out_w])
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

/// Generates a fractional pooling sequence.
///
/// Creates a sequence of indices that divides input_size into output_size regions.
fn generate_fractional_sequence(input_size: usize, output_size: usize) -> Vec<usize> {
    let mut sequence = Vec::with_capacity(output_size + 1);
    let alpha = input_size as f64 / output_size as f64;

    for i in 0..=output_size {
        let idx = (alpha * i as f64).round() as usize;
        sequence.push(idx.min(input_size));
    }

    // Ensure last element is input_size
    if let Some(last) = sequence.last_mut() {
        *last = input_size;
    }

    sequence
}
