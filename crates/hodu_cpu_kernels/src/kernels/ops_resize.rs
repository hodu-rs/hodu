//! Resize operations for tensors
//!
//! This module provides image/tensor resizing with various interpolation modes:
//! - Nearest: Nearest neighbor interpolation
//! - Linear: Bilinear (2D) / Trilinear (3D) interpolation
//! - Cubic: Bicubic interpolation (2D only)
//!
//! Supports ONNX-compatible coordinate transformation modes:
//! - HalfPixel (default)
//! - Asymmetric
//! - AlignCorners
//! - PytorchHalfPixel

use crate::{
    error::Result,
    kernels::{macros::ops, Kernel},
};
use core::ffi::c_void;

ops!(resize);

extern "C" {
    fn hodu_cpu_resize_f8e4m3(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_resize_f8e5m2(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_resize_bf16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_resize_f16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_resize_f32(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_resize_f64(input: *const c_void, output: *mut c_void, metadata: *const usize);
}

/// Call resize operation by kernel name
///
/// Resizes spatial dimensions of the input tensor using various interpolation modes.
///
/// # Metadata layout
/// - metadata[0]: output_size (total number of elements in output)
/// - metadata[1]: num_dims (number of dimensions, typically 4 for NCHW or 5 for NCDHW)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: input_strides
/// - metadata[2+2*num_dims]: offset (starting offset in input)
/// - metadata[3+2*num_dims..3+3*num_dims]: output_shape
/// - metadata[3+3*num_dims]: mode (0=nearest, 1=linear, 2=cubic)
/// - metadata[4+3*num_dims]: coord_transform (0=half_pixel, 1=asymmetric, 2=align_corners, 3=pytorch_half_pixel)
/// - metadata[5+3*num_dims]: nearest_mode (0=floor, 1=ceil, 2=round_prefer_floor, 3=round_prefer_ceil)
///
/// # Interpolation modes
/// - Nearest (0): Nearest neighbor, no interpolation
/// - Linear (1): Bilinear for 2D spatial dims, trilinear for 3D
/// - Cubic (2): Bicubic interpolation (2D only)
///
/// # Coordinate transformation modes
/// - HalfPixel (0): out_coord = (in_coord + 0.5) * scale - 0.5
/// - Asymmetric (1): out_coord = in_coord * scale
/// - AlignCorners (2): out_coord = in_coord * (in_size - 1) / (out_size - 1)
/// - PytorchHalfPixel (3): Like HalfPixel but returns 0 when output size is 1
///
/// # Safety
/// - `input` must point to valid tensor data of the appropriate type
/// - `output` must point to a valid output buffer with sufficient capacity
/// - Metadata must accurately describe the tensor layout and resize parameters
pub fn call_ops_resize(
    kernel_name: Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    let kernel_str = kernel_name.0;
    unsafe {
        match kernel_str {
            "hodu_cpu_resize_f8e4m3" => hodu_cpu_resize_f8e4m3(input, output, metadata.as_ptr()),
            "hodu_cpu_resize_f8e5m2" => hodu_cpu_resize_f8e5m2(input, output, metadata.as_ptr()),
            "hodu_cpu_resize_bf16" => hodu_cpu_resize_bf16(input, output, metadata.as_ptr()),
            "hodu_cpu_resize_f16" => hodu_cpu_resize_f16(input, output, metadata.as_ptr()),
            "hodu_cpu_resize_f32" => hodu_cpu_resize_f32(input, output, metadata.as_ptr()),
            "hodu_cpu_resize_f64" => hodu_cpu_resize_f64(input, output, metadata.as_ptr()),
            _ => panic!("Unsupported resize kernel: {:?}", kernel_name),
        }
    }

    Ok(())
}
