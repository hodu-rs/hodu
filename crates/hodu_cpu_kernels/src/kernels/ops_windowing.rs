//! Windowing operations for tensors
//!
//! This module provides sliding window reduction operations:
//! - reduce_window_max: Maximum value in each window
//! - reduce_window_min: Minimum value in each window
//! - reduce_window_sum: Sum of values in each window
//! - reduce_window_mean: Mean (average) of values in each window
//!
//! These operations apply a reduction function over sliding windows of the input tensor,
//! with support for configurable window size, stride, and padding.

use crate::{
    error::Result,
    kernels::{macros::ops, Kernel},
};
use core::ffi::c_void;

ops!(
    reduce_window_max,
    reduce_window_mean,
    reduce_window_sum,
    reduce_window_min
);

extern "C" {
    fn hodu_cpu_reduce_window_max_f8e4m3(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_max_f8e5m2(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_max_bf16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_max_f16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_max_f32(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_max_f64(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_max_i8(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_max_i16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_max_i32(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_max_i64(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_max_u8(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_max_u16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_max_u32(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_max_u64(input: *const c_void, output: *mut c_void, metadata: *const usize);

    fn hodu_cpu_reduce_window_mean_f8e4m3(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_mean_f8e5m2(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_mean_bf16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_mean_f16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_mean_f32(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_mean_f64(input: *const c_void, output: *mut c_void, metadata: *const usize);

    fn hodu_cpu_reduce_window_sum_f8e4m3(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_sum_f8e5m2(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_sum_bf16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_sum_f16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_sum_f32(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_sum_f64(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_sum_i8(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_sum_i16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_sum_i32(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_sum_i64(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_sum_u8(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_sum_u16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_sum_u32(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_sum_u64(input: *const c_void, output: *mut c_void, metadata: *const usize);

    fn hodu_cpu_reduce_window_min_f8e4m3(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_min_f8e5m2(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_min_bf16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_min_f16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_min_f32(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_min_f64(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_min_i8(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_min_i16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_min_i32(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_min_i64(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_min_u8(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_min_u16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_min_u32(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_reduce_window_min_u64(input: *const c_void, output: *mut c_void, metadata: *const usize);
}

/// Call windowing reduction operation by kernel name
///
/// Applies a reduction operation over sliding windows of the input tensor.
///
/// # Metadata layout
/// - metadata[0]: output_size (total number of elements in output)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: input_strides
/// - metadata[2+2*num_dims]: offset (starting offset in input)
/// - metadata[3+2*num_dims..3+3*num_dims]: window_shape (size of window in each dimension)
/// - metadata[3+3*num_dims..3+4*num_dims]: strides (step size in each dimension)
/// - metadata[3+4*num_dims..3+4*num_dims+2*num_dims]: padding (before and after for each dimension)
/// - metadata[3+6*num_dims..]: output_shape
///
/// # Padding layout
/// For each dimension i, padding is specified as [pad_before_i, pad_after_i].
/// Padded areas are treated as negative infinity for max, positive infinity for min,
/// and zero for sum/mean operations.
///
/// # Safety
/// - `input` must point to valid tensor data of the appropriate type
/// - `output` must point to a valid output buffer with sufficient capacity
/// - Metadata must accurately describe the tensor layout and window parameters
pub fn call_ops_reduce_window(
    kernel_name: Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    let kernel_str = kernel_name.0;
    unsafe {
        match kernel_str {
            "hodu_cpu_reduce_window_max_f8e4m3" => hodu_cpu_reduce_window_max_f8e4m3(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_max_f8e5m2" => hodu_cpu_reduce_window_max_f8e5m2(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_max_bf16" => hodu_cpu_reduce_window_max_bf16(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_max_f16" => hodu_cpu_reduce_window_max_f16(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_max_f32" => hodu_cpu_reduce_window_max_f32(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_max_f64" => hodu_cpu_reduce_window_max_f64(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_max_i8" => hodu_cpu_reduce_window_max_i8(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_max_i16" => hodu_cpu_reduce_window_max_i16(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_max_i32" => hodu_cpu_reduce_window_max_i32(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_max_i64" => hodu_cpu_reduce_window_max_i64(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_max_u8" => hodu_cpu_reduce_window_max_u8(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_max_u16" => hodu_cpu_reduce_window_max_u16(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_max_u32" => hodu_cpu_reduce_window_max_u32(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_max_u64" => hodu_cpu_reduce_window_max_u64(input, output, metadata.as_ptr()),

            "hodu_cpu_reduce_window_mean_f8e4m3" => {
                hodu_cpu_reduce_window_mean_f8e4m3(input, output, metadata.as_ptr())
            },
            "hodu_cpu_reduce_window_mean_f8e5m2" => {
                hodu_cpu_reduce_window_mean_f8e5m2(input, output, metadata.as_ptr())
            },
            "hodu_cpu_reduce_window_mean_bf16" => hodu_cpu_reduce_window_mean_bf16(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_mean_f16" => hodu_cpu_reduce_window_mean_f16(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_mean_f32" => hodu_cpu_reduce_window_mean_f32(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_mean_f64" => hodu_cpu_reduce_window_mean_f64(input, output, metadata.as_ptr()),

            "hodu_cpu_reduce_window_sum_f8e4m3" => hodu_cpu_reduce_window_sum_f8e4m3(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_sum_f8e5m2" => hodu_cpu_reduce_window_sum_f8e5m2(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_sum_bf16" => hodu_cpu_reduce_window_sum_bf16(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_sum_f16" => hodu_cpu_reduce_window_sum_f16(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_sum_f32" => hodu_cpu_reduce_window_sum_f32(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_sum_f64" => hodu_cpu_reduce_window_sum_f64(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_sum_i8" => hodu_cpu_reduce_window_sum_i8(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_sum_i16" => hodu_cpu_reduce_window_sum_i16(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_sum_i32" => hodu_cpu_reduce_window_sum_i32(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_sum_i64" => hodu_cpu_reduce_window_sum_i64(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_sum_u8" => hodu_cpu_reduce_window_sum_u8(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_sum_u16" => hodu_cpu_reduce_window_sum_u16(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_sum_u32" => hodu_cpu_reduce_window_sum_u32(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_sum_u64" => hodu_cpu_reduce_window_sum_u64(input, output, metadata.as_ptr()),

            "hodu_cpu_reduce_window_min_f8e4m3" => hodu_cpu_reduce_window_min_f8e4m3(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_min_f8e5m2" => hodu_cpu_reduce_window_min_f8e5m2(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_min_bf16" => hodu_cpu_reduce_window_min_bf16(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_min_f16" => hodu_cpu_reduce_window_min_f16(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_min_f32" => hodu_cpu_reduce_window_min_f32(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_min_f64" => hodu_cpu_reduce_window_min_f64(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_min_i8" => hodu_cpu_reduce_window_min_i8(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_min_i16" => hodu_cpu_reduce_window_min_i16(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_min_i32" => hodu_cpu_reduce_window_min_i32(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_min_i64" => hodu_cpu_reduce_window_min_i64(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_min_u8" => hodu_cpu_reduce_window_min_u8(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_min_u16" => hodu_cpu_reduce_window_min_u16(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_min_u32" => hodu_cpu_reduce_window_min_u32(input, output, metadata.as_ptr()),
            "hodu_cpu_reduce_window_min_u64" => hodu_cpu_reduce_window_min_u64(input, output, metadata.as_ptr()),

            _ => panic!("Unsupported reduce_window kernel: {:?}", kernel_name),
        }
    }

    Ok(())
}
