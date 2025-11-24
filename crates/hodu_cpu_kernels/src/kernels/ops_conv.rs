//! Convolution operations for neural network layers
//!
//! This module provides:
//! - Standard convolutions (conv1d, conv2d, conv3d)
//! - Transposed convolutions (conv_transpose1d, conv_transpose2d, conv_transpose3d)
//! - Gradient weight computations for backpropagation
//!
//! All operations support padding, stride, and dilation parameters.

use crate::{
    error::Result,
    kernels::macros::{ops, Kernel},
};
use core::ffi::c_void;

ops!(
    conv1d,
    conv2d,
    conv3d,
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
    conv1d_grad_weight,
    conv2d_grad_weight,
    conv3d_grad_weight,
    conv_transpose1d_grad_weight,
    conv_transpose2d_grad_weight,
    conv_transpose3d_grad_weight
);

extern "C" {
    fn hodu_cpu_conv1d_f8e4m3(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv1d_f8e5m2(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv1d_bf16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv1d_f16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv1d_f32(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv1d_f64(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv2d_f8e4m3(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv2d_f8e5m2(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv2d_bf16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv2d_f16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv2d_f32(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv2d_f64(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv3d_f8e4m3(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv3d_f8e5m2(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv3d_bf16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv3d_f16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv3d_f32(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv3d_f64(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_conv_transpose1d_f8e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose1d_f8e5m2(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose1d_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose1d_f16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose1d_f32(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose1d_f64(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose2d_f8e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose2d_f8e5m2(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose2d_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose2d_f16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose2d_f32(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose2d_f64(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose3d_f8e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose3d_f8e5m2(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose3d_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose3d_f16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose3d_f32(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose3d_f64(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv1d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv1d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv1d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv1d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv1d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv1d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv2d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv2d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv2d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv2d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv2d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv2d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv3d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv3d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv3d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv3d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv3d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv3d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose1d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose1d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose1d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose1d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose1d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose1d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose2d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose2d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose2d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose2d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose2d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose2d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose3d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose3d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose3d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose3d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose3d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn hodu_cpu_conv_transpose3d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
}

/// Execute a convolution operation
///
/// Performs forward pass of convolution operations including standard and transposed convolutions.
/// Supports 1D, 2D, and 3D spatial dimensions with configurable padding, stride, and dilation.
///
/// # Arguments
/// * `kernel` - The convolution kernel to execute (e.g., conv2d::F32, conv_transpose1d::F64)
/// * `input` - Pointer to input tensor data
/// * `weight` - Pointer to convolution weight/kernel tensor
/// * `output` - Pointer to output tensor buffer
/// * `metadata` - Tensor metadata array (see layouts below)
///
/// # Metadata layout for conv1d / conv_transpose1d
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: batch
/// - metadata[2]: in_channels
/// - metadata[3]: out_channels
/// - metadata[4]: in_width
/// - metadata[5]: kernel_width
/// - metadata[6]: out_width
/// - metadata[7]: stride
/// - metadata[8]: padding
/// - metadata[9]: dilation
/// - metadata[10]: input_offset
/// - metadata[11]: weight_offset
///
/// # Metadata layout for conv2d / conv_transpose2d
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: batch
/// - metadata[2]: in_channels
/// - metadata[3]: out_channels
/// - metadata[4]: in_height
/// - metadata[5]: in_width
/// - metadata[6]: kernel_height
/// - metadata[7]: kernel_width
/// - metadata[8]: out_height
/// - metadata[9]: out_width
/// - metadata[10]: stride_h
/// - metadata[11]: stride_w
/// - metadata[12]: padding_h
/// - metadata[13]: padding_w
/// - metadata[14]: dilation_h
/// - metadata[15]: dilation_w
/// - metadata[16]: input_offset
/// - metadata[17]: weight_offset
///
/// # Metadata layout for conv3d / conv_transpose3d
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: batch
/// - metadata[2]: in_channels
/// - metadata[3]: out_channels
/// - metadata[4]: in_depth
/// - metadata[5]: in_height
/// - metadata[6]: in_width
/// - metadata[7]: kernel_depth
/// - metadata[8]: kernel_height
/// - metadata[9]: kernel_width
/// - metadata[10]: out_depth
/// - metadata[11]: out_height
/// - metadata[12]: out_width
/// - metadata[13]: stride_d
/// - metadata[14]: stride_h
/// - metadata[15]: stride_w
/// - metadata[16]: padding_d
/// - metadata[17]: padding_h
/// - metadata[18]: padding_w
/// - metadata[19]: dilation_d
/// - metadata[20]: dilation_h
/// - metadata[21]: dilation_w
/// - metadata[22]: input_offset
/// - metadata[23]: weight_offset
///
/// # Safety
/// This function uses unsafe FFI calls to C kernels. Caller must ensure:
/// - All pointers are valid and properly aligned
/// - Metadata accurately describes tensor layout and convolution parameters
/// - Output buffer has sufficient capacity
///
/// # Returns
/// Returns `Ok(())` on success.
pub fn call_ops_conv(
    kernel: Kernel,
    input: *const c_void,
    weight: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        match kernel {
            conv1d::F8E4M3 => hodu_cpu_conv1d_f8e4m3(input, weight, output, metadata.as_ptr()),
            conv1d::F8E5M2 => hodu_cpu_conv1d_f8e5m2(input, weight, output, metadata.as_ptr()),
            conv1d::BF16 => hodu_cpu_conv1d_bf16(input, weight, output, metadata.as_ptr()),
            conv1d::F16 => hodu_cpu_conv1d_f16(input, weight, output, metadata.as_ptr()),
            conv1d::F32 => hodu_cpu_conv1d_f32(input, weight, output, metadata.as_ptr()),
            conv1d::F64 => hodu_cpu_conv1d_f64(input, weight, output, metadata.as_ptr()),
            conv2d::F8E4M3 => hodu_cpu_conv2d_f8e4m3(input, weight, output, metadata.as_ptr()),
            conv2d::F8E5M2 => hodu_cpu_conv2d_f8e5m2(input, weight, output, metadata.as_ptr()),
            conv2d::BF16 => hodu_cpu_conv2d_bf16(input, weight, output, metadata.as_ptr()),
            conv2d::F16 => hodu_cpu_conv2d_f16(input, weight, output, metadata.as_ptr()),
            conv2d::F32 => hodu_cpu_conv2d_f32(input, weight, output, metadata.as_ptr()),
            conv2d::F64 => hodu_cpu_conv2d_f64(input, weight, output, metadata.as_ptr()),
            conv3d::F8E4M3 => hodu_cpu_conv3d_f8e4m3(input, weight, output, metadata.as_ptr()),
            conv3d::F8E5M2 => hodu_cpu_conv3d_f8e5m2(input, weight, output, metadata.as_ptr()),
            conv3d::BF16 => hodu_cpu_conv3d_bf16(input, weight, output, metadata.as_ptr()),
            conv3d::F16 => hodu_cpu_conv3d_f16(input, weight, output, metadata.as_ptr()),
            conv3d::F32 => hodu_cpu_conv3d_f32(input, weight, output, metadata.as_ptr()),
            conv3d::F64 => hodu_cpu_conv3d_f64(input, weight, output, metadata.as_ptr()),
            conv_transpose1d::F8E4M3 => hodu_cpu_conv_transpose1d_f8e4m3(input, weight, output, metadata.as_ptr()),
            conv_transpose1d::F8E5M2 => hodu_cpu_conv_transpose1d_f8e5m2(input, weight, output, metadata.as_ptr()),
            conv_transpose1d::BF16 => hodu_cpu_conv_transpose1d_bf16(input, weight, output, metadata.as_ptr()),
            conv_transpose1d::F16 => hodu_cpu_conv_transpose1d_f16(input, weight, output, metadata.as_ptr()),
            conv_transpose1d::F32 => hodu_cpu_conv_transpose1d_f32(input, weight, output, metadata.as_ptr()),
            conv_transpose1d::F64 => hodu_cpu_conv_transpose1d_f64(input, weight, output, metadata.as_ptr()),
            conv_transpose2d::F8E4M3 => hodu_cpu_conv_transpose2d_f8e4m3(input, weight, output, metadata.as_ptr()),
            conv_transpose2d::F8E5M2 => hodu_cpu_conv_transpose2d_f8e5m2(input, weight, output, metadata.as_ptr()),
            conv_transpose2d::BF16 => hodu_cpu_conv_transpose2d_bf16(input, weight, output, metadata.as_ptr()),
            conv_transpose2d::F16 => hodu_cpu_conv_transpose2d_f16(input, weight, output, metadata.as_ptr()),
            conv_transpose2d::F32 => hodu_cpu_conv_transpose2d_f32(input, weight, output, metadata.as_ptr()),
            conv_transpose2d::F64 => hodu_cpu_conv_transpose2d_f64(input, weight, output, metadata.as_ptr()),
            conv_transpose3d::F8E4M3 => hodu_cpu_conv_transpose3d_f8e4m3(input, weight, output, metadata.as_ptr()),
            conv_transpose3d::F8E5M2 => hodu_cpu_conv_transpose3d_f8e5m2(input, weight, output, metadata.as_ptr()),
            conv_transpose3d::BF16 => hodu_cpu_conv_transpose3d_bf16(input, weight, output, metadata.as_ptr()),
            conv_transpose3d::F16 => hodu_cpu_conv_transpose3d_f16(input, weight, output, metadata.as_ptr()),
            conv_transpose3d::F32 => hodu_cpu_conv_transpose3d_f32(input, weight, output, metadata.as_ptr()),
            conv_transpose3d::F64 => hodu_cpu_conv_transpose3d_f64(input, weight, output, metadata.as_ptr()),
            _ => panic!("Unsupported conv kernel: {:?}", kernel),
        }
    }

    Ok(())
}

/// Execute a convolution gradient weight operation
///
/// Computes gradients with respect to convolution weights during backpropagation.
/// Used for training neural networks with convolutional layers.
///
/// # Arguments
/// * `kernel` - The gradient kernel to execute (e.g., conv2d_grad_weight::F32)
/// * `input` - Pointer to input tensor data (forward pass input)
/// * `grad_output` - Pointer to gradient tensor from next layer
/// * `grad_weight` - Pointer to output gradient weight buffer
/// * `metadata` - Tensor metadata array (same layout as forward pass)
///
/// # Metadata layout (Generic, dimension-agnostic)
///
/// All grad_weight operations use a unified metadata structure:
///
/// - metadata[0]: num_els (total grad_weight elements)
/// - metadata[1]: input_ndim
/// - metadata[2]: spatial_dims
/// - metadata[3..3+input_ndim]: input_shape
/// - metadata[3+input_ndim..3+2*input_ndim]: grad_output_shape
/// - metadata[3+2*input_ndim..3+3*input_ndim]: weight_shape
/// - metadata[3+3*input_ndim..3+4*input_ndim]: input_strides
/// - metadata[3+4*input_ndim..3+5*input_ndim]: grad_output_strides
/// - metadata[3+5*input_ndim]: input_offset
/// - metadata[3+5*input_ndim+1]: grad_output_offset
/// - metadata[3+5*input_ndim+2..]: stride, padding, dilation (spatial_dims elements each)
///
/// ## Examples:
///
/// Conv1D (input_ndim=3, spatial_dims=1):
/// - metadata[18]: input_offset, metadata[19]: grad_output_offset
/// - metadata[20]: stride, metadata[21]: padding, metadata[22]: dilation
///
/// Conv2D (input_ndim=4, spatial_dims=2):
/// - metadata[23]: input_offset, metadata[24]: grad_output_offset
/// - metadata[25..27]: stride, metadata[27..29]: padding, metadata[29..31]: dilation
///
/// Conv3D (input_ndim=5, spatial_dims=3):
/// - metadata[28]: input_offset, metadata[29]: grad_output_offset
/// - metadata[30..33]: stride, metadata[33..36]: padding, metadata[36..39]: dilation
///
/// Transpose convolutions use the same layout.
///
/// # Safety
/// This function uses unsafe FFI calls to C kernels. Caller must ensure:
/// - All pointers are valid and properly aligned
/// - Metadata accurately describes tensor layout
/// - grad_weight buffer has sufficient capacity and is initialized to zero
///
/// # Returns
/// Returns `Ok(())` on success.
pub fn call_ops_conv_grad_weight(
    kernel: Kernel,
    input: *const c_void,
    grad_output: *const c_void,
    grad_weight: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        match kernel {
            conv1d_grad_weight::F8E4M3 => {
                hodu_cpu_conv1d_grad_weight_f8e4m3(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv1d_grad_weight::F8E5M2 => {
                hodu_cpu_conv1d_grad_weight_f8e5m2(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv1d_grad_weight::BF16 => {
                hodu_cpu_conv1d_grad_weight_bf16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv1d_grad_weight::F16 => {
                hodu_cpu_conv1d_grad_weight_f16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv1d_grad_weight::F32 => {
                hodu_cpu_conv1d_grad_weight_f32(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv1d_grad_weight::F64 => {
                hodu_cpu_conv1d_grad_weight_f64(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv2d_grad_weight::F8E4M3 => {
                hodu_cpu_conv2d_grad_weight_f8e4m3(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv2d_grad_weight::F8E5M2 => {
                hodu_cpu_conv2d_grad_weight_f8e5m2(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv2d_grad_weight::BF16 => {
                hodu_cpu_conv2d_grad_weight_bf16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv2d_grad_weight::F16 => {
                hodu_cpu_conv2d_grad_weight_f16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv2d_grad_weight::F32 => {
                hodu_cpu_conv2d_grad_weight_f32(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv2d_grad_weight::F64 => {
                hodu_cpu_conv2d_grad_weight_f64(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv3d_grad_weight::F8E4M3 => {
                hodu_cpu_conv3d_grad_weight_f8e4m3(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv3d_grad_weight::F8E5M2 => {
                hodu_cpu_conv3d_grad_weight_f8e5m2(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv3d_grad_weight::BF16 => {
                hodu_cpu_conv3d_grad_weight_bf16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv3d_grad_weight::F16 => {
                hodu_cpu_conv3d_grad_weight_f16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv3d_grad_weight::F32 => {
                hodu_cpu_conv3d_grad_weight_f32(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv3d_grad_weight::F64 => {
                hodu_cpu_conv3d_grad_weight_f64(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::F8E4M3 => {
                hodu_cpu_conv_transpose1d_grad_weight_f8e4m3(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::F8E5M2 => {
                hodu_cpu_conv_transpose1d_grad_weight_f8e5m2(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::BF16 => {
                hodu_cpu_conv_transpose1d_grad_weight_bf16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::F16 => {
                hodu_cpu_conv_transpose1d_grad_weight_f16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::F32 => {
                hodu_cpu_conv_transpose1d_grad_weight_f32(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::F64 => {
                hodu_cpu_conv_transpose1d_grad_weight_f64(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::F8E4M3 => {
                hodu_cpu_conv_transpose2d_grad_weight_f8e4m3(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::F8E5M2 => {
                hodu_cpu_conv_transpose2d_grad_weight_f8e5m2(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::BF16 => {
                hodu_cpu_conv_transpose2d_grad_weight_bf16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::F16 => {
                hodu_cpu_conv_transpose2d_grad_weight_f16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::F32 => {
                hodu_cpu_conv_transpose2d_grad_weight_f32(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::F64 => {
                hodu_cpu_conv_transpose2d_grad_weight_f64(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::F8E4M3 => {
                hodu_cpu_conv_transpose3d_grad_weight_f8e4m3(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::F8E5M2 => {
                hodu_cpu_conv_transpose3d_grad_weight_f8e5m2(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::BF16 => {
                hodu_cpu_conv_transpose3d_grad_weight_bf16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::F16 => {
                hodu_cpu_conv_transpose3d_grad_weight_f16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::F32 => {
                hodu_cpu_conv_transpose3d_grad_weight_f32(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::F64 => {
                hodu_cpu_conv_transpose3d_grad_weight_f64(input, grad_output, grad_weight, metadata.as_ptr())
            },
            _ => panic!("Unsupported conv grad_weight kernel: {:?}", kernel),
        }
    }

    Ok(())
}
