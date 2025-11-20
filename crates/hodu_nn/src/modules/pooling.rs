use crate::compat::*;
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

    fn parameters(&mut self) -> Vec<&mut Tensor> {
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

    fn parameters(&mut self) -> Vec<&mut Tensor> {
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

    fn parameters(&mut self) -> Vec<&mut Tensor> {
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

    fn parameters(&mut self) -> Vec<&mut Tensor> {
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

    fn parameters(&mut self) -> Vec<&mut Tensor> {
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

    fn parameters(&mut self) -> Vec<&mut Tensor> {
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

    fn parameters(&mut self) -> Vec<&mut Tensor> {
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

    fn parameters(&mut self) -> Vec<&mut Tensor> {
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

    fn parameters(&mut self) -> Vec<&mut Tensor> {
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

    fn parameters(&mut self) -> Vec<&mut Tensor> {
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

    fn parameters(&mut self) -> Vec<&mut Tensor> {
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

    fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}
