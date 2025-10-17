use crate::compat::*;
use crate::module::Module;
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor, types::dtype::DType};

#[derive(Module, Clone)]
pub struct Conv1D {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv1D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        with_bias: bool,
        dtype: DType,
    ) -> HoduResult<Self> {
        // Kaiming initialization for Conv layers
        let k: f32 = (2.0 / (in_channels * kernel_size) as f32).sqrt();

        let zero = Scalar::zero(dtype);
        let one = Scalar::one(dtype);
        let k_scalar = Scalar::from_f32(k, dtype);

        // weight: [out_channels, in_channels, kernel_size]
        let weight = Tensor::randn(&[out_channels, in_channels, kernel_size], zero, one)?;
        weight.set_requires_grad(true)?;
        let weight = weight.mul(&Tensor::full(&[], k_scalar)?)?;

        // bias: [out_channels]
        let bias = if with_bias {
            let bias = Tensor::randn(&[out_channels], zero, one)?;
            bias.set_requires_grad(true)?;
            let bias = bias.mul(&Tensor::full(&[], k_scalar)?)?;
            Some(bias)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
        })
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // Conv1D: input [N, Ci, L], weight [Co, Ci, K]
        let output = input.conv1d(&self.weight, self.stride, self.padding, self.dilation)?;

        if let Some(ref bias) = self.bias {
            // Add bias: output [N, Co, L_out], bias [Co]
            // Need to reshape bias to [1, Co, 1] for broadcasting
            let bias_layout = bias.get_layout();
            let bias_shape = bias_layout.get_shape();
            let bias_reshaped = bias.reshape(&[1, bias_shape[0], 1])?;
            output.add(&bias_reshaped)
        } else {
            Ok(output)
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
}

#[derive(Module, Clone)]
pub struct Conv2D {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        with_bias: bool,
        dtype: DType,
    ) -> HoduResult<Self> {
        // Kaiming initialization for Conv layers
        let k: f32 = (2.0 / (in_channels * kernel_size * kernel_size) as f32).sqrt();

        let zero = Scalar::zero(dtype);
        let one = Scalar::one(dtype);
        let k_scalar = Scalar::from_f32(k, dtype);

        // weight: [out_channels, in_channels, kernel_size, kernel_size]
        let weight = Tensor::randn(&[out_channels, in_channels, kernel_size, kernel_size], zero, one)?;
        weight.set_requires_grad(true)?;
        let weight = weight.mul(&Tensor::full(&[], k_scalar)?)?;

        // bias: [out_channels]
        let bias = if with_bias {
            let bias = Tensor::randn(&[out_channels], zero, one)?;
            bias.set_requires_grad(true)?;
            let bias = bias.mul(&Tensor::full(&[], k_scalar)?)?;
            Some(bias)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
        })
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // Conv2D: input [N, Ci, H, W], weight [Co, Ci, Kh, Kw]
        let output = input.conv2d(&self.weight, self.stride, self.padding, self.dilation)?;

        if let Some(ref bias) = self.bias {
            // Add bias: output [N, Co, H_out, W_out], bias [Co]
            // Need to reshape bias to [1, Co, 1, 1] for broadcasting
            let bias_layout = bias.get_layout();
            let bias_shape = bias_layout.get_shape();
            let bias_reshaped = bias.reshape(&[1, bias_shape[0], 1, 1])?;
            output.add(&bias_reshaped)
        } else {
            Ok(output)
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
}

#[derive(Module, Clone)]
pub struct Conv3D {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv3D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        with_bias: bool,
        dtype: DType,
    ) -> HoduResult<Self> {
        // Kaiming initialization for Conv layers
        let k: f32 = (2.0 / (in_channels * kernel_size * kernel_size * kernel_size) as f32).sqrt();

        let zero = Scalar::zero(dtype);
        let one = Scalar::one(dtype);
        let k_scalar = Scalar::from_f32(k, dtype);

        // weight: [out_channels, in_channels, kernel_size, kernel_size, kernel_size]
        let weight = Tensor::randn(
            &[out_channels, in_channels, kernel_size, kernel_size, kernel_size],
            zero,
            one,
        )?;
        weight.set_requires_grad(true)?;
        let weight = weight.mul(&Tensor::full(&[], k_scalar)?)?;

        // bias: [out_channels]
        let bias = if with_bias {
            let bias = Tensor::randn(&[out_channels], zero, one)?;
            bias.set_requires_grad(true)?;
            let bias = bias.mul(&Tensor::full(&[], k_scalar)?)?;
            Some(bias)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
        })
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // Conv3D: input [N, Ci, D, H, W], weight [Co, Ci, Kd, Kh, Kw]
        let output = input.conv3d(&self.weight, self.stride, self.padding, self.dilation)?;

        if let Some(ref bias) = self.bias {
            // Add bias: output [N, Co, D_out, H_out, W_out], bias [Co]
            // Need to reshape bias to [1, Co, 1, 1, 1] for broadcasting
            let bias_layout = bias.get_layout();
            let bias_shape = bias_layout.get_shape();
            let bias_reshaped = bias.reshape(&[1, bias_shape[0], 1, 1, 1])?;
            output.add(&bias_reshaped)
        } else {
            Ok(output)
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
}

#[derive(Module, Clone)]
pub struct ConvTranspose1D {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
}

impl ConvTranspose1D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        with_bias: bool,
        dtype: DType,
    ) -> HoduResult<Self> {
        // Kaiming initialization for ConvTranspose layers
        let k: f32 = (2.0 / (in_channels * kernel_size) as f32).sqrt();

        let zero = Scalar::zero(dtype);
        let one = Scalar::one(dtype);
        let k_scalar = Scalar::from_f32(k, dtype);

        // weight: [in_channels, out_channels, kernel_size]
        let weight = Tensor::randn(&[in_channels, out_channels, kernel_size], zero, one)?;
        weight.set_requires_grad(true)?;
        let weight = weight.mul(&Tensor::full(&[], k_scalar)?)?;

        // bias: [out_channels]
        let bias = if with_bias {
            let bias = Tensor::randn(&[out_channels], zero, one)?;
            bias.set_requires_grad(true)?;
            let bias = bias.mul(&Tensor::full(&[], k_scalar)?)?;
            Some(bias)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
        })
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // ConvTranspose1D: input [N, Ci, L], weight [Ci, Co, K]
        let output = input.conv_transpose1d(
            &self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )?;

        if let Some(ref bias) = self.bias {
            // Add bias: output [N, Co, L_out], bias [Co]
            // Need to reshape bias to [1, Co, 1] for broadcasting
            let bias_layout = bias.get_layout();
            let bias_shape = bias_layout.get_shape();
            let bias_reshaped = bias.reshape(&[1, bias_shape[0], 1])?;
            output.add(&bias_reshaped)
        } else {
            Ok(output)
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
}

#[derive(Module, Clone)]
pub struct ConvTranspose2D {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
}

impl ConvTranspose2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        with_bias: bool,
        dtype: DType,
    ) -> HoduResult<Self> {
        // Kaiming initialization for ConvTranspose layers
        let k: f32 = (2.0 / (in_channels * kernel_size * kernel_size) as f32).sqrt();

        let zero = Scalar::zero(dtype);
        let one = Scalar::one(dtype);
        let k_scalar = Scalar::from_f32(k, dtype);

        // weight: [in_channels, out_channels, kernel_size, kernel_size]
        let weight = Tensor::randn(&[in_channels, out_channels, kernel_size, kernel_size], zero, one)?;
        weight.set_requires_grad(true)?;
        let weight = weight.mul(&Tensor::full(&[], k_scalar)?)?;

        // bias: [out_channels]
        let bias = if with_bias {
            let bias = Tensor::randn(&[out_channels], zero, one)?;
            bias.set_requires_grad(true)?;
            let bias = bias.mul(&Tensor::full(&[], k_scalar)?)?;
            Some(bias)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
        })
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // ConvTranspose2D: input [N, Ci, H, W], weight [Ci, Co, Kh, Kw]
        let output = input.conv_transpose2d(
            &self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )?;

        if let Some(ref bias) = self.bias {
            // Add bias: output [N, Co, H_out, W_out], bias [Co]
            // Need to reshape bias to [1, Co, 1, 1] for broadcasting
            let bias_layout = bias.get_layout();
            let bias_shape = bias_layout.get_shape();
            let bias_reshaped = bias.reshape(&[1, bias_shape[0], 1, 1])?;
            output.add(&bias_reshaped)
        } else {
            Ok(output)
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
}

#[derive(Module, Clone)]
pub struct ConvTranspose3D {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
}

impl ConvTranspose3D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        with_bias: bool,
        dtype: DType,
    ) -> HoduResult<Self> {
        // Kaiming initialization for ConvTranspose layers
        let k: f32 = (2.0 / (in_channels * kernel_size * kernel_size * kernel_size) as f32).sqrt();

        let zero = Scalar::zero(dtype);
        let one = Scalar::one(dtype);
        let k_scalar = Scalar::from_f32(k, dtype);

        // weight: [in_channels, out_channels, kernel_size, kernel_size, kernel_size]
        let weight = Tensor::randn(
            &[in_channels, out_channels, kernel_size, kernel_size, kernel_size],
            zero,
            one,
        )?;
        weight.set_requires_grad(true)?;
        let weight = weight.mul(&Tensor::full(&[], k_scalar)?)?;

        // bias: [out_channels]
        let bias = if with_bias {
            let bias = Tensor::randn(&[out_channels], zero, one)?;
            bias.set_requires_grad(true)?;
            let bias = bias.mul(&Tensor::full(&[], k_scalar)?)?;
            Some(bias)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
        })
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // ConvTranspose3D: input [N, Ci, D, H, W], weight [Ci, Co, Kd, Kh, Kw]
        let output = input.conv_transpose3d(
            &self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )?;

        if let Some(ref bias) = self.bias {
            // Add bias: output [N, Co, D_out, H_out, W_out], bias [Co]
            // Need to reshape bias to [1, Co, 1, 1, 1] for broadcasting
            let bias_layout = bias.get_layout();
            let bias_shape = bias_layout.get_shape();
            let bias_reshaped = bias.reshape(&[1, bias_shape[0], 1, 1, 1])?;
            output.add(&bias_reshaped)
        } else {
            Ok(output)
        }
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
}
