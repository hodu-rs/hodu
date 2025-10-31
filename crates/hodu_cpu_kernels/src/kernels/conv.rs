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
    fn conv1d_f8e4m3(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv1d_f8e5m2(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv1d_bf16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv1d_f16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv1d_f32(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv1d_f64(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv2d_f8e4m3(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv2d_f8e5m2(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv2d_bf16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv2d_f16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv2d_f32(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv2d_f64(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv3d_f8e4m3(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv3d_f8e5m2(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv3d_bf16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv3d_f16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv3d_f32(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv3d_f64(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv_transpose1d_f8e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose1d_f8e5m2(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose1d_bf16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv_transpose1d_f16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv_transpose1d_f32(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv_transpose1d_f64(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv_transpose2d_f8e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose2d_f8e5m2(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose2d_bf16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv_transpose2d_f16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv_transpose2d_f32(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv_transpose2d_f64(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv_transpose3d_f8e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose3d_f8e5m2(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose3d_bf16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv_transpose3d_f16(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv_transpose3d_f32(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv_transpose3d_f64(input: *const c_void, weight: *const c_void, output: *mut c_void, metadata: *const usize);
    fn conv1d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv1d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv1d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv1d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv1d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv1d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv2d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv2d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv2d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv2d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv2d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv2d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv3d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv3d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv3d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv3d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv3d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv3d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose1d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose1d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose1d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose1d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose1d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose1d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose2d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose2d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose2d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose2d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose2d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose2d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose3d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose3d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose3d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose3d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose3d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        metadata: *const usize,
    );
    fn conv_transpose3d_grad_weight_f64(
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
/// - metadata[1]: in_channels
/// - metadata[2]: out_channels
/// - metadata[3]: in_width
/// - metadata[4]: kernel_width
/// - metadata[5]: out_width
/// - metadata[6]: stride
/// - metadata[7]: padding
/// - metadata[8]: dilation
/// - metadata[9]: input_offset
/// - metadata[10]: weight_offset
///
/// # Metadata layout for conv2d / conv_transpose2d
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: in_channels
/// - metadata[2]: out_channels
/// - metadata[3]: in_height
/// - metadata[4]: in_width
/// - metadata[5]: kernel_height
/// - metadata[6]: kernel_width
/// - metadata[7]: out_height
/// - metadata[8]: out_width
/// - metadata[9]: stride_h
/// - metadata[10]: stride_w
/// - metadata[11]: padding_h
/// - metadata[12]: padding_w
/// - metadata[13]: dilation_h
/// - metadata[14]: dilation_w
/// - metadata[15]: input_offset
/// - metadata[16]: weight_offset
///
/// # Metadata layout for conv3d / conv_transpose3d
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: in_channels
/// - metadata[2]: out_channels
/// - metadata[3]: in_depth
/// - metadata[4]: in_height
/// - metadata[5]: in_width
/// - metadata[6]: kernel_depth
/// - metadata[7]: kernel_height
/// - metadata[8]: kernel_width
/// - metadata[9]: out_depth
/// - metadata[10]: out_height
/// - metadata[11]: out_width
/// - metadata[12]: stride_d
/// - metadata[13]: stride_h
/// - metadata[14]: stride_w
/// - metadata[15]: padding_d
/// - metadata[16]: padding_h
/// - metadata[17]: padding_w
/// - metadata[18]: dilation_d
/// - metadata[19]: dilation_h
/// - metadata[20]: dilation_w
/// - metadata[21]: input_offset
/// - metadata[22]: weight_offset
///
/// # Safety
/// This function uses unsafe FFI calls to C kernels. Caller must ensure:
/// - All pointers are valid and properly aligned
/// - Metadata accurately describes tensor layout and convolution parameters
/// - Output buffer has sufficient capacity
///
/// # Returns
/// Returns `Ok(())` on success.
pub fn call_conv(
    kernel: Kernel,
    input: *const c_void,
    weight: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        match kernel {
            conv1d::F8E4M3 => conv1d_f8e4m3(input, weight, output, metadata.as_ptr()),
            conv1d::F8E5M2 => conv1d_f8e5m2(input, weight, output, metadata.as_ptr()),
            conv1d::BF16 => conv1d_bf16(input, weight, output, metadata.as_ptr()),
            conv1d::F16 => conv1d_f16(input, weight, output, metadata.as_ptr()),
            conv1d::F32 => conv1d_f32(input, weight, output, metadata.as_ptr()),
            conv1d::F64 => conv1d_f64(input, weight, output, metadata.as_ptr()),
            conv2d::F8E4M3 => conv2d_f8e4m3(input, weight, output, metadata.as_ptr()),
            conv2d::F8E5M2 => conv2d_f8e5m2(input, weight, output, metadata.as_ptr()),
            conv2d::BF16 => conv2d_bf16(input, weight, output, metadata.as_ptr()),
            conv2d::F16 => conv2d_f16(input, weight, output, metadata.as_ptr()),
            conv2d::F32 => conv2d_f32(input, weight, output, metadata.as_ptr()),
            conv2d::F64 => conv2d_f64(input, weight, output, metadata.as_ptr()),
            conv3d::F8E4M3 => conv3d_f8e4m3(input, weight, output, metadata.as_ptr()),
            conv3d::F8E5M2 => conv3d_f8e5m2(input, weight, output, metadata.as_ptr()),
            conv3d::BF16 => conv3d_bf16(input, weight, output, metadata.as_ptr()),
            conv3d::F16 => conv3d_f16(input, weight, output, metadata.as_ptr()),
            conv3d::F32 => conv3d_f32(input, weight, output, metadata.as_ptr()),
            conv3d::F64 => conv3d_f64(input, weight, output, metadata.as_ptr()),
            conv_transpose1d::F8E4M3 => conv_transpose1d_f8e4m3(input, weight, output, metadata.as_ptr()),
            conv_transpose1d::F8E5M2 => conv_transpose1d_f8e5m2(input, weight, output, metadata.as_ptr()),
            conv_transpose1d::BF16 => conv_transpose1d_bf16(input, weight, output, metadata.as_ptr()),
            conv_transpose1d::F16 => conv_transpose1d_f16(input, weight, output, metadata.as_ptr()),
            conv_transpose1d::F32 => conv_transpose1d_f32(input, weight, output, metadata.as_ptr()),
            conv_transpose1d::F64 => conv_transpose1d_f64(input, weight, output, metadata.as_ptr()),
            conv_transpose2d::F8E4M3 => conv_transpose2d_f8e4m3(input, weight, output, metadata.as_ptr()),
            conv_transpose2d::F8E5M2 => conv_transpose2d_f8e5m2(input, weight, output, metadata.as_ptr()),
            conv_transpose2d::BF16 => conv_transpose2d_bf16(input, weight, output, metadata.as_ptr()),
            conv_transpose2d::F16 => conv_transpose2d_f16(input, weight, output, metadata.as_ptr()),
            conv_transpose2d::F32 => conv_transpose2d_f32(input, weight, output, metadata.as_ptr()),
            conv_transpose2d::F64 => conv_transpose2d_f64(input, weight, output, metadata.as_ptr()),
            conv_transpose3d::F8E4M3 => conv_transpose3d_f8e4m3(input, weight, output, metadata.as_ptr()),
            conv_transpose3d::F8E5M2 => conv_transpose3d_f8e5m2(input, weight, output, metadata.as_ptr()),
            conv_transpose3d::BF16 => conv_transpose3d_bf16(input, weight, output, metadata.as_ptr()),
            conv_transpose3d::F16 => conv_transpose3d_f16(input, weight, output, metadata.as_ptr()),
            conv_transpose3d::F32 => conv_transpose3d_f32(input, weight, output, metadata.as_ptr()),
            conv_transpose3d::F64 => conv_transpose3d_f64(input, weight, output, metadata.as_ptr()),
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
/// # Metadata layout
/// Same as `call_conv` - see that function for complete metadata layouts for 1D, 2D, and 3D variants.
/// Note: metadata[1] is used as batch_size in gradient computation.
///
/// # Safety
/// This function uses unsafe FFI calls to C kernels. Caller must ensure:
/// - All pointers are valid and properly aligned
/// - Metadata accurately describes tensor layout
/// - grad_weight buffer has sufficient capacity and is initialized to zero
///
/// # Returns
/// Returns `Ok(())` on success.
pub fn call_conv_grad_weight(
    kernel: Kernel,
    input: *const c_void,
    grad_output: *const c_void,
    grad_weight: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        match kernel {
            conv1d_grad_weight::F8E4M3 => conv1d_grad_weight_f8e4m3(input, grad_output, grad_weight, metadata.as_ptr()),
            conv1d_grad_weight::F8E5M2 => conv1d_grad_weight_f8e5m2(input, grad_output, grad_weight, metadata.as_ptr()),
            conv1d_grad_weight::BF16 => conv1d_grad_weight_bf16(input, grad_output, grad_weight, metadata.as_ptr()),
            conv1d_grad_weight::F16 => conv1d_grad_weight_f16(input, grad_output, grad_weight, metadata.as_ptr()),
            conv1d_grad_weight::F32 => conv1d_grad_weight_f32(input, grad_output, grad_weight, metadata.as_ptr()),
            conv1d_grad_weight::F64 => conv1d_grad_weight_f64(input, grad_output, grad_weight, metadata.as_ptr()),
            conv2d_grad_weight::F8E4M3 => conv2d_grad_weight_f8e4m3(input, grad_output, grad_weight, metadata.as_ptr()),
            conv2d_grad_weight::F8E5M2 => conv2d_grad_weight_f8e5m2(input, grad_output, grad_weight, metadata.as_ptr()),
            conv2d_grad_weight::BF16 => conv2d_grad_weight_bf16(input, grad_output, grad_weight, metadata.as_ptr()),
            conv2d_grad_weight::F16 => conv2d_grad_weight_f16(input, grad_output, grad_weight, metadata.as_ptr()),
            conv2d_grad_weight::F32 => conv2d_grad_weight_f32(input, grad_output, grad_weight, metadata.as_ptr()),
            conv2d_grad_weight::F64 => conv2d_grad_weight_f64(input, grad_output, grad_weight, metadata.as_ptr()),
            conv3d_grad_weight::F8E4M3 => conv3d_grad_weight_f8e4m3(input, grad_output, grad_weight, metadata.as_ptr()),
            conv3d_grad_weight::F8E5M2 => conv3d_grad_weight_f8e5m2(input, grad_output, grad_weight, metadata.as_ptr()),
            conv3d_grad_weight::BF16 => conv3d_grad_weight_bf16(input, grad_output, grad_weight, metadata.as_ptr()),
            conv3d_grad_weight::F16 => conv3d_grad_weight_f16(input, grad_output, grad_weight, metadata.as_ptr()),
            conv3d_grad_weight::F32 => conv3d_grad_weight_f32(input, grad_output, grad_weight, metadata.as_ptr()),
            conv3d_grad_weight::F64 => conv3d_grad_weight_f64(input, grad_output, grad_weight, metadata.as_ptr()),
            conv_transpose1d_grad_weight::F8E4M3 => {
                conv_transpose1d_grad_weight_f8e4m3(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::F8E5M2 => {
                conv_transpose1d_grad_weight_f8e5m2(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::BF16 => {
                conv_transpose1d_grad_weight_bf16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::F16 => {
                conv_transpose1d_grad_weight_f16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::F32 => {
                conv_transpose1d_grad_weight_f32(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::F64 => {
                conv_transpose1d_grad_weight_f64(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::F8E4M3 => {
                conv_transpose2d_grad_weight_f8e4m3(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::F8E5M2 => {
                conv_transpose2d_grad_weight_f8e5m2(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::BF16 => {
                conv_transpose2d_grad_weight_bf16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::F16 => {
                conv_transpose2d_grad_weight_f16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::F32 => {
                conv_transpose2d_grad_weight_f32(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::F64 => {
                conv_transpose2d_grad_weight_f64(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::F8E4M3 => {
                conv_transpose3d_grad_weight_f8e4m3(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::F8E5M2 => {
                conv_transpose3d_grad_weight_f8e5m2(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::BF16 => {
                conv_transpose3d_grad_weight_bf16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::F16 => {
                conv_transpose3d_grad_weight_f16(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::F32 => {
                conv_transpose3d_grad_weight_f32(input, grad_output, grad_weight, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::F64 => {
                conv_transpose3d_grad_weight_f64(input, grad_output, grad_weight, metadata.as_ptr())
            },
            _ => panic!("Unsupported conv grad_weight kernel: {:?}", kernel),
        }
    }

    Ok(())
}
