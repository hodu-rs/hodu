use crate::{error::Result, kernels::macros::ops};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::ffi::c_void;

ops!(concat, split);

// Macro to automatically generate extern declarations and dispatch for concat operations
macro_rules! declare_and_dispatch_concat {
    ($($op:ident),* $(,)?) => {
        paste::paste! {
            // Extern C declarations for concat operations
            extern "C" {
                $(
                    fn [<$op _bool>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f8e4m3>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f8e5m2>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _bf16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i8>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u8>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                )*
            }

            // Dispatch function for concat operations
            unsafe fn dispatch_concat(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                num_els: usize,
                num_dims: usize,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!(stringify!($op), "_bool") => [<$op _bool>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f8e4m3") => [<$op _f8e4m3>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f8e5m2") => [<$op _f8e5m2>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_bf16") => [<$op _bf16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f16") => [<$op _f16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f32") => [<$op _f32>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f64") => [<$op _f64>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i8") => [<$op _i8>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i16") => [<$op _i16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i32") => [<$op _i32>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i64") => [<$op _i64>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u8") => [<$op _u8>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u16") => [<$op _u16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u32") => [<$op _u32>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u64") => [<$op _u64>](input, output, num_els, num_dims, metadata),
                    )*
                    _ => panic!("Unknown concat operation: {}", name),
                }
            }
        }
    };
}

// Macro to automatically generate extern declarations and dispatch for split operations
macro_rules! declare_and_dispatch_split {
    ($($op:ident),* $(,)?) => {
        paste::paste! {
            // Extern C declarations for split operations
            extern "C" {
                $(
                    fn [<$op _bool>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f8e4m3>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f8e5m2>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _bf16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i8>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u8>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                )*
            }

            // Dispatch function for split operations
            unsafe fn dispatch_split(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                num_els: usize,
                num_dims: usize,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!(stringify!($op), "_bool") => [<$op _bool>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f8e4m3") => [<$op _f8e4m3>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f8e5m2") => [<$op _f8e5m2>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_bf16") => [<$op _bf16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f16") => [<$op _f16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f32") => [<$op _f32>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f64") => [<$op _f64>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i8") => [<$op _i8>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i16") => [<$op _i16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i32") => [<$op _i32>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i64") => [<$op _i64>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u8") => [<$op _u8>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u16") => [<$op _u16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u32") => [<$op _u32>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u64") => [<$op _u64>](input, output, num_els, num_dims, metadata),
                    )*
                    _ => panic!("Unknown split operation: {}", name),
                }
            }
        }
    };
}

declare_and_dispatch_concat!(concat);
declare_and_dispatch_split!(split);

/// Call concat operation - concatenate multiple tensors along a dimension
#[allow(clippy::too_many_arguments)]
pub fn call_concat(
    kernel_name: crate::kernels::macros::Kernel,
    output_shape: &[usize],
    concat_dim: usize,
    input_shapes: &[usize],
    input_strides: &[usize],
    input_offsets: &[usize],
    input_buffer_offsets: &[usize],
    input: *const c_void,
    output: *mut c_void,
) -> Result<()> {
    let num_dims = output_shape.len();
    let num_els: usize = output_shape.iter().product();
    let num_inputs = input_offsets.len();

    // Prepare metadata: output_shape, concat_dim, num_inputs, input_shapes, input_strides, input_offsets, input_buffer_offsets
    let mut metadata = Vec::with_capacity(
        num_dims + 2 + input_shapes.len() + input_strides.len() + input_offsets.len() + input_buffer_offsets.len(),
    );
    metadata.extend_from_slice(output_shape);
    metadata.push(concat_dim);
    metadata.push(num_inputs);
    metadata.extend_from_slice(input_shapes);
    metadata.extend_from_slice(input_strides);
    metadata.extend_from_slice(input_offsets);
    metadata.extend_from_slice(input_buffer_offsets);

    unsafe {
        dispatch_concat(kernel_name.0, input, output, num_els, num_dims, metadata.as_ptr());
    }

    Ok(())
}

/// Call split operation - split tensor into multiple outputs along a dimension
#[allow(clippy::too_many_arguments)]
pub fn call_split(
    kernel_name: crate::kernels::macros::Kernel,
    input_shape: &[usize],
    input: *const c_void,
    input_strides: &[usize],
    input_offset: usize,
    split_dim: usize,
    output_size_on_dim: usize,
    split_offset: usize,
    output: *mut c_void,
) -> Result<()> {
    let num_dims = input_shape.len();

    // Calculate output shape
    let mut output_shape = input_shape.to_vec();
    output_shape[split_dim] = output_size_on_dim;
    let num_els: usize = output_shape.iter().product();

    // Prepare metadata: input_shape, input_strides, input_offset, split_dim, output_size_on_dim, split_offset
    let mut metadata = Vec::with_capacity(num_dims * 2 + 4);
    metadata.extend_from_slice(input_shape);
    metadata.extend_from_slice(input_strides);
    metadata.push(input_offset);
    metadata.push(split_dim);
    metadata.push(output_size_on_dim);
    metadata.push(split_offset);

    unsafe {
        dispatch_split(kernel_name.0, input, output, num_els, num_dims, metadata.as_ptr());
    }

    Ok(())
}
