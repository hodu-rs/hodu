//! Concatenation and split operations for tensors
//!
//! This module provides operations to concatenate multiple tensors along a dimension
//! or split a single tensor into multiple outputs along a dimension.

use crate::{error::Result, kernels::macros::ops};
use core::ffi::c_void;

ops!(concat, split);

/// Macro to generate extern C declarations and dispatch logic for concat operations
///
/// This macro generates FFI bindings for all supported data types.
macro_rules! declare_and_dispatch_concat {
    ($($op:ident),* $(,)?) => {
        paste::paste! {
            // Extern C declarations for concat operations
            extern "C" {
                $(
                    fn [<$op _bool>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _f8e4m3>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _f8e5m2>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _bf16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _f16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _f32>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _f64>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _i8>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _i16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _i32>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _i64>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _u8>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _u16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _u32>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _u64>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                )*
            }

            // Dispatch function for concat operations
            unsafe fn dispatch_concat(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!(stringify!($op), "_bool") => [<$op _bool>](input, output, metadata),
                        concat!(stringify!($op), "_f8e4m3") => [<$op _f8e4m3>](input, output, metadata),
                        concat!(stringify!($op), "_f8e5m2") => [<$op _f8e5m2>](input, output, metadata),
                        concat!(stringify!($op), "_bf16") => [<$op _bf16>](input, output, metadata),
                        concat!(stringify!($op), "_f16") => [<$op _f16>](input, output, metadata),
                        concat!(stringify!($op), "_f32") => [<$op _f32>](input, output, metadata),
                        concat!(stringify!($op), "_f64") => [<$op _f64>](input, output, metadata),
                        concat!(stringify!($op), "_i8") => [<$op _i8>](input, output, metadata),
                        concat!(stringify!($op), "_i16") => [<$op _i16>](input, output, metadata),
                        concat!(stringify!($op), "_i32") => [<$op _i32>](input, output, metadata),
                        concat!(stringify!($op), "_i64") => [<$op _i64>](input, output, metadata),
                        concat!(stringify!($op), "_u8") => [<$op _u8>](input, output, metadata),
                        concat!(stringify!($op), "_u16") => [<$op _u16>](input, output, metadata),
                        concat!(stringify!($op), "_u32") => [<$op _u32>](input, output, metadata),
                        concat!(stringify!($op), "_u64") => [<$op _u64>](input, output, metadata),
                    )*
                    _ => panic!("Unknown concat operation: {}", name),
                }
            }
        }
    };
}

/// Macro to generate extern C declarations and dispatch logic for split operations
///
/// This macro generates FFI bindings for all supported data types.
macro_rules! declare_and_dispatch_split {
    ($($op:ident),* $(,)?) => {
        paste::paste! {
            // Extern C declarations for split operations
            extern "C" {
                $(
                    fn [<$op _bool>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _f8e4m3>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _f8e5m2>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _bf16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _f16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _f32>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _f64>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _i8>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _i16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _i32>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _i64>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _u8>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _u16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _u32>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _u64>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                )*
            }

            // Dispatch function for split operations
            unsafe fn dispatch_split(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!(stringify!($op), "_bool") => [<$op _bool>](input, output, metadata),
                        concat!(stringify!($op), "_f8e4m3") => [<$op _f8e4m3>](input, output, metadata),
                        concat!(stringify!($op), "_f8e5m2") => [<$op _f8e5m2>](input, output, metadata),
                        concat!(stringify!($op), "_bf16") => [<$op _bf16>](input, output, metadata),
                        concat!(stringify!($op), "_f16") => [<$op _f16>](input, output, metadata),
                        concat!(stringify!($op), "_f32") => [<$op _f32>](input, output, metadata),
                        concat!(stringify!($op), "_f64") => [<$op _f64>](input, output, metadata),
                        concat!(stringify!($op), "_i8") => [<$op _i8>](input, output, metadata),
                        concat!(stringify!($op), "_i16") => [<$op _i16>](input, output, metadata),
                        concat!(stringify!($op), "_i32") => [<$op _i32>](input, output, metadata),
                        concat!(stringify!($op), "_i64") => [<$op _i64>](input, output, metadata),
                        concat!(stringify!($op), "_u8") => [<$op _u8>](input, output, metadata),
                        concat!(stringify!($op), "_u16") => [<$op _u16>](input, output, metadata),
                        concat!(stringify!($op), "_u32") => [<$op _u32>](input, output, metadata),
                        concat!(stringify!($op), "_u64") => [<$op _u64>](input, output, metadata),
                    )*
                    _ => panic!("Unknown split operation: {}", name),
                }
            }
        }
    };
}

declare_and_dispatch_concat!(concat);
declare_and_dispatch_split!(split);

/// Execute a concatenation operation on multiple tensors
///
/// Concatenates multiple input tensors along a specified dimension. All input tensors
/// must have the same shape except along the concatenation dimension. The input buffer
/// contains all input tensors stored contiguously, with offsets specified in metadata.
///
/// # Arguments
/// * `kernel_name` - The concat operation kernel (e.g., concat::F32)
/// * `input` - Pointer to input buffer containing all input tensors
/// * `output` - Pointer to output tensor buffer
/// * `metadata` - Tensor metadata array (see layout below)
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements in output)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: output_shape (shape of concatenated output)
/// - metadata[2+num_dims]: concat_dim (dimension along which to concatenate)
/// - metadata[2+num_dims+1]: num_inputs (number of input tensors)
/// - metadata[2+num_dims+2..2+num_dims+2+num_inputs*num_dims]: input_shapes (flattened: [input0_shape..., input1_shape..., ...])
/// - metadata[...+num_inputs*num_dims]: input_strides (flattened: [input0_strides..., input1_strides..., ...])
/// - metadata[...+num_inputs]: input_offsets (offset within each input tensor: [offset0, offset1, ...])
/// - metadata[...+num_inputs]: input_buffer_offsets (byte offset in input buffer: [buf_offset0, buf_offset1, ...])
///
/// # Safety
/// This function uses unsafe FFI calls to C kernels. Caller must ensure:
/// - Input buffer contains all input tensors at specified buffer offsets
/// - All pointers are valid and properly aligned
/// - Metadata accurately describes tensor layout
/// - Output buffer has sufficient capacity
///
/// # Returns
/// Returns `Ok(())` on success.
pub fn call_ops_concat(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_concat(kernel_name.0, input, output, metadata.as_ptr());
    }

    Ok(())
}

/// Execute a split operation on a tensor
///
/// Extracts a slice from an input tensor along a specified dimension. This operation
/// is the inverse of concatenation and can be used to extract one of multiple tensors
/// that were previously concatenated.
///
/// # Arguments
/// * `kernel_name` - The split operation kernel (e.g., split::F32)
/// * `input` - Pointer to input tensor data
/// * `output` - Pointer to output tensor buffer
/// * `metadata` - Tensor metadata array (see layout below)
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements in output)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape (shape of input tensor)
/// - metadata[2+num_dims..2+2*num_dims]: input_strides (strides of input tensor)
/// - metadata[2+2*num_dims]: input_offset (starting offset in input tensor)
/// - metadata[2+2*num_dims+1]: split_dim (dimension along which to split)
/// - metadata[2+2*num_dims+2]: output_size_on_dim (size of output along split dimension)
/// - metadata[2+2*num_dims+3]: split_offset (offset along split dimension where output starts)
///
/// # Safety
/// This function uses unsafe FFI calls to C kernels. Caller must ensure:
/// - Input pointer is valid and properly aligned
/// - Metadata accurately describes tensor layout
/// - Output buffer has sufficient capacity
/// - split_offset + output_size_on_dim <= input_shape[split_dim]
///
/// # Returns
/// Returns `Ok(())` on success.
pub fn call_ops_split(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_split(kernel_name.0, input, output, metadata.as_ptr());
    }

    Ok(())
}
