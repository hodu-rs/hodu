//! Padding operations
//!
//! This module provides padding operations for tensors:
//! - pad_constant: Pad with a constant value
//! - pad_reflect: Pad with reflected values at boundaries
//! - pad_replicate: Pad by replicating edge values
//! - pad_circular: Pad with circular/wrapped values
//!
//! All operations support strided tensor access and multiple data types.

use crate::{error::Result, kernels::macros::ops};
use core::ffi::c_void;

ops!(pad_constant, pad_reflect, pad_replicate, pad_circular);

/// Call pad_constant operation by kernel name
///
/// Pads tensor with a constant value.
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: output_shape
/// - metadata[2+2*num_dims..2+3*num_dims]: pad_before (padding before each dim)
///
/// # Safety
/// - `input` must point to valid tensor data of the appropriate type
/// - `output` must point to a valid output buffer with sufficient capacity
/// - `pad_value` must point to a single value of the appropriate type
/// - Metadata must accurately describe the tensor layout
pub fn call_ops_pad_constant(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    pad_value: *const c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_pad_constant(kernel_name.0, input, output, pad_value, metadata.as_ptr());
    }
    Ok(())
}

/// Call pad_reflect operation by kernel name
///
/// Pads tensor with reflected values at boundaries.
/// For input [1, 2, 3] with pad=2: [3, 2, 1, 2, 3, 2, 1]
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: output_shape
/// - metadata[2+2*num_dims..2+3*num_dims]: pad_before (padding before each dim)
///
/// # Safety
/// - `input` must point to valid tensor data of the appropriate type
/// - `output` must point to a valid output buffer with sufficient capacity
/// - Metadata must accurately describe the tensor layout
pub fn call_ops_pad_reflect(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_pad_reflect(kernel_name.0, input, output, metadata.as_ptr());
    }
    Ok(())
}

/// Call pad_replicate operation by kernel name
///
/// Pads tensor by replicating edge values.
/// For input [1, 2, 3] with pad=2: [1, 1, 1, 2, 3, 3, 3]
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: output_shape
/// - metadata[2+2*num_dims..2+3*num_dims]: pad_before (padding before each dim)
///
/// # Safety
/// - `input` must point to valid tensor data of the appropriate type
/// - `output` must point to a valid output buffer with sufficient capacity
/// - Metadata must accurately describe the tensor layout
pub fn call_ops_pad_replicate(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_pad_replicate(kernel_name.0, input, output, metadata.as_ptr());
    }
    Ok(())
}

/// Call pad_circular operation by kernel name
///
/// Pads tensor with circular/wrapped values.
/// For input [1, 2, 3] with pad=2: [2, 3, 1, 2, 3, 1, 2]
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: output_shape
/// - metadata[2+2*num_dims..2+3*num_dims]: pad_before (padding before each dim)
///
/// # Safety
/// - `input` must point to valid tensor data of the appropriate type
/// - `output` must point to a valid output buffer with sufficient capacity
/// - Metadata must accurately describe the tensor layout
pub fn call_ops_pad_circular(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_pad_circular(kernel_name.0, input, output, metadata.as_ptr());
    }
    Ok(())
}

macro_rules! declare_and_dispatch_pad_constant {
    ($($type_suffix:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<hodu_cpu_pad_constant_ $type_suffix>](
                        input: *const c_void,
                        output: *mut c_void,
                        pad_value: *const c_void,
                        metadata: *const usize
                    );
                )*
            }

            unsafe fn dispatch_pad_constant(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                pad_value: *const c_void,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!("hodu_cpu_pad_constant_", stringify!($type_suffix)) => {
                            [<hodu_cpu_pad_constant_ $type_suffix>](input, output, pad_value, metadata)
                        }
                    )*
                    _ => panic!("Unsupported pad_constant kernel: {}", name),
                }
            }
        }
    };
}

macro_rules! declare_and_dispatch_pad_reflect {
    ($($type_suffix:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<hodu_cpu_pad_reflect_ $type_suffix>](
                        input: *const c_void,
                        output: *mut c_void,
                        metadata: *const usize
                    );
                )*
            }

            unsafe fn dispatch_pad_reflect(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!("hodu_cpu_pad_reflect_", stringify!($type_suffix)) => {
                            [<hodu_cpu_pad_reflect_ $type_suffix>](input, output, metadata)
                        }
                    )*
                    _ => panic!("Unsupported pad_reflect kernel: {}", name),
                }
            }
        }
    };
}

macro_rules! declare_and_dispatch_pad_replicate {
    ($($type_suffix:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<hodu_cpu_pad_replicate_ $type_suffix>](
                        input: *const c_void,
                        output: *mut c_void,
                        metadata: *const usize
                    );
                )*
            }

            unsafe fn dispatch_pad_replicate(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!("hodu_cpu_pad_replicate_", stringify!($type_suffix)) => {
                            [<hodu_cpu_pad_replicate_ $type_suffix>](input, output, metadata)
                        }
                    )*
                    _ => panic!("Unsupported pad_replicate kernel: {}", name),
                }
            }
        }
    };
}

macro_rules! declare_and_dispatch_pad_circular {
    ($($type_suffix:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<hodu_cpu_pad_circular_ $type_suffix>](
                        input: *const c_void,
                        output: *mut c_void,
                        metadata: *const usize
                    );
                )*
            }

            unsafe fn dispatch_pad_circular(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!("hodu_cpu_pad_circular_", stringify!($type_suffix)) => {
                            [<hodu_cpu_pad_circular_ $type_suffix>](input, output, metadata)
                        }
                    )*
                    _ => panic!("Unsupported pad_circular kernel: {}", name),
                }
            }
        }
    };
}

declare_and_dispatch_pad_constant!(bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
declare_and_dispatch_pad_reflect!(bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
declare_and_dispatch_pad_replicate!(bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
declare_and_dispatch_pad_circular!(bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
