//! Reduction operations for aggregating tensor values along dimensions
//!
//! This module provides various reduction operations:
//! - Aggregations: `sum`, `mean`, `prod`
//! - Statistics: `std`, `var`, `norm`
//! - Extrema: `max`, `min`
//! - Indices: `argmax`, `argmin`
//! - Logical: `any`, `all`
//!
//! All operations support multi-dimensional reductions with configurable dimensions
//! and optional dimension preservation (keep_dim).

use crate::{error::Result, kernels::macros::ops};
use core::ffi::c_void;

// Define all reduce operations using the macro
ops!(sum, mean, max, min, prod, std, var, norm, argmax, argmin, any, all);

/// Execute a reduction operation
///
/// Reduces input tensor along specified dimensions using the given reduction operation.
/// The reduction can preserve dimensions (keep_dim=true) or squeeze them (keep_dim=false).
///
/// # Arguments
/// * `kernel_name` - The reduction kernel to execute (e.g., sum::F32, mean::F64)
/// * `input` - Pointer to input tensor data
/// * `output` - Pointer to output tensor buffer
/// * `metadata` - Tensor metadata array (see layout below)
///
/// # Metadata layout
/// - metadata[0]: shape_len (number of dimensions in input)
/// - metadata[1..1+shape_len]: shape (shape of input tensor)
/// - metadata[1+shape_len..1+2*shape_len]: strides (strides of input tensor)
/// - metadata[1+2*shape_len]: offset (starting offset in input)
/// - metadata[2+2*shape_len]: output_shape_len (number of dimensions in output)
/// - metadata[3+2*shape_len..3+2*shape_len+output_shape_len]: output_shape (shape of output tensor)
/// - metadata[3+2*shape_len+output_shape_len]: num_reduce_dims (number of dimensions to reduce)
/// - metadata[4+2*shape_len+output_shape_len..4+2*shape_len+output_shape_len+num_reduce_dims]: reduce_dims (dimension indices to reduce)
/// - metadata[4+2*shape_len+output_shape_len+num_reduce_dims]: keep_dim (1 to keep dimensions as size 1, 0 to squeeze)
/// - metadata[5+2*shape_len+output_shape_len+num_reduce_dims]: reduce_size (total number of elements to reduce per output element)
///
/// # Operation-specific behaviors
/// - `sum`, `mean`, `prod`: Available for all numeric types
/// - `std`, `var`, `norm`, `mean`: Float types only
/// - `max`, `min`: Available for all numeric types
/// - `argmax`, `argmin`: Return int32 indices, available for all types including bool
/// - `any`, `all`: Return bool, available for all types including bool
///
/// # Safety
/// This function uses unsafe FFI calls to C kernels. Caller must ensure:
/// - All pointers are valid and properly aligned
/// - Metadata accurately describes tensor layout
/// - Output buffer has sufficient capacity
/// - reduce_dims contains valid dimension indices
///
/// # Returns
/// Returns `Ok(())` on success.
pub fn call_reduce(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_reduce(kernel_name.0, input, output, metadata.as_ptr());
    }

    Ok(())
}

/// Macro to declare extern C functions and dispatch for reduce operations
///
/// This macro generates FFI bindings for reduction operations with different type support:
/// - all_types: Supports all numeric types (sum, max, min, prod)
/// - float_types: Supports only floating point types (mean, std, var, norm)
/// - all_and_bool_types: Supports all types including bool (argmax, argmin, any, all)
macro_rules! declare_reduce_ops {
    // All types (sum, max, min, prod)
    (all_types: $($op:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
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
        }
    };

    // Float types only (mean, norm, std, var)
    (float_types: $($op:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<$op _f8e4m3>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _f8e5m2>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _bf16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _f16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _f32>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<$op _f64>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                )*
            }
        }
    };

    // All and bool types
    (all_and_bool_types: $($op:ident),* $(,)?) => {
        paste::paste! {
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
        }
    };
}

// Declare all reduce operations
declare_reduce_ops!(all_types: sum, max, min, prod);
declare_reduce_ops!(float_types: std, var, mean, norm);
declare_reduce_ops!(all_and_bool_types: argmax, argmin, any, all);

// Macro to generate dispatch match arms
macro_rules! dispatch_all_types {
    ($name:expr, $input:expr, $output:expr, $metadata:expr, $($op:ident),* $(,)?) => {
        paste::paste! {
            match $name {
                $(
                    concat!(stringify!($op), "_f8e4m3") => [<$op _f8e4m3>]($input, $output, $metadata),
                    concat!(stringify!($op), "_f8e5m2") => [<$op _f8e5m2>]($input, $output, $metadata),
                    concat!(stringify!($op), "_bf16") => [<$op _bf16>]($input, $output, $metadata),
                    concat!(stringify!($op), "_f16") => [<$op _f16>]($input, $output, $metadata),
                    concat!(stringify!($op), "_f32") => [<$op _f32>]($input, $output, $metadata),
                    concat!(stringify!($op), "_f64") => [<$op _f64>]($input, $output, $metadata),
                    concat!(stringify!($op), "_i8") => [<$op _i8>]($input, $output, $metadata),
                    concat!(stringify!($op), "_i16") => [<$op _i16>]($input, $output, $metadata),
                    concat!(stringify!($op), "_i32") => [<$op _i32>]($input, $output, $metadata),
                    concat!(stringify!($op), "_i64") => [<$op _i64>]($input, $output, $metadata),
                    concat!(stringify!($op), "_u8") => [<$op _u8>]($input, $output, $metadata),
                    concat!(stringify!($op), "_u16") => [<$op _u16>]($input, $output, $metadata),
                    concat!(stringify!($op), "_u32") => [<$op _u32>]($input, $output, $metadata),
                    concat!(stringify!($op), "_u64") => [<$op _u64>]($input, $output, $metadata),
                )*
                _ => {}
            }
        }
    };
}

macro_rules! dispatch_float_types {
    ($name:expr, $input:expr, $output:expr, $metadata:expr, $($op:ident),* $(,)?) => {
        paste::paste! {
            match $name {
                $(
                    concat!(stringify!($op), "_f8e4m3") => [<$op _f8e4m3>]($input, $output, $metadata),
                    concat!(stringify!($op), "_f8e5m2") => [<$op _f8e5m2>]($input, $output, $metadata),
                    concat!(stringify!($op), "_bf16") => [<$op _bf16>]($input, $output, $metadata),
                    concat!(stringify!($op), "_f16") => [<$op _f16>]($input, $output, $metadata),
                    concat!(stringify!($op), "_f32") => [<$op _f32>]($input, $output, $metadata),
                    concat!(stringify!($op), "_f64") => [<$op _f64>]($input, $output, $metadata),
                )*
                _ => {}
            }
        }
    };
}

macro_rules! dispatch_all_and_bool_types {
    ($name:expr, $input:expr, $output:expr, $metadata:expr, $($op:ident),* $(,)?) => {
        paste::paste! {
            match $name {
                $(
                    concat!(stringify!($op), "_bool") => [<$op _bool>]($input, $output, $metadata),
                    concat!(stringify!($op), "_f8e4m3") => [<$op _f8e4m3>]($input, $output, $metadata),
                    concat!(stringify!($op), "_f8e5m2") => [<$op _f8e5m2>]($input, $output, $metadata),
                    concat!(stringify!($op), "_bf16") => [<$op _bf16>]($input, $output, $metadata),
                    concat!(stringify!($op), "_f16") => [<$op _f16>]($input, $output, $metadata),
                    concat!(stringify!($op), "_f32") => [<$op _f32>]($input, $output, $metadata),
                    concat!(stringify!($op), "_f64") => [<$op _f64>]($input, $output, $metadata),
                    concat!(stringify!($op), "_i8") => [<$op _i8>]($input, $output, $metadata),
                    concat!(stringify!($op), "_i16") => [<$op _i16>]($input, $output, $metadata),
                    concat!(stringify!($op), "_i32") => [<$op _i32>]($input, $output, $metadata),
                    concat!(stringify!($op), "_i64") => [<$op _i64>]($input, $output, $metadata),
                    concat!(stringify!($op), "_u8") => [<$op _u8>]($input, $output, $metadata),
                    concat!(stringify!($op), "_u16") => [<$op _u16>]($input, $output, $metadata),
                    concat!(stringify!($op), "_u32") => [<$op _u32>]($input, $output, $metadata),
                    concat!(stringify!($op), "_u64") => [<$op _u64>]($input, $output, $metadata),
                )*
                _ => {}
            }
        }
    };
}

// Dispatch function for reduce operations
unsafe fn dispatch_reduce(name: &str, input: *const c_void, output: *mut c_void, metadata: *const usize) {
    dispatch_all_types!(name, input, output, metadata, sum, max, min, prod);

    dispatch_float_types!(name, input, output, metadata, std, var, mean, norm);

    dispatch_all_and_bool_types!(name, input, output, metadata, argmax, argmin, any, all);

    // If we get here, kernel was not found
    if !matches!(
        name,
        "sum_f8e4m3"
            | "sum_f8e5m2"
            | "sum_bf16"
            | "sum_f16"
            | "sum_f32"
            | "sum_f64"
            | "sum_i8"
            | "sum_i16"
            | "sum_i32"
            | "sum_i64"
            | "sum_u8"
            | "sum_u16"
            | "sum_u32"
            | "sum_u64"
            | "mean_f8e4m3"
            | "mean_f8e5m2"
            | "mean_bf16"
            | "mean_f16"
            | "mean_f32"
            | "mean_f64"
            | "max_f8e4m3"
            | "max_f8e5m2"
            | "max_bf16"
            | "max_f16"
            | "max_f32"
            | "max_f64"
            | "max_i8"
            | "max_i16"
            | "max_i32"
            | "max_i64"
            | "max_u8"
            | "max_u16"
            | "max_u32"
            | "max_u64"
            | "min_f8e4m3"
            | "min_f8e5m2"
            | "min_bf16"
            | "min_f16"
            | "min_f32"
            | "min_f64"
            | "min_i8"
            | "min_i16"
            | "min_i32"
            | "min_i64"
            | "min_u8"
            | "min_u16"
            | "min_u32"
            | "min_u64"
            | "prod_f8e4m3"
            | "prod_f8e5m2"
            | "prod_bf16"
            | "prod_f16"
            | "prod_f32"
            | "prod_f64"
            | "prod_i8"
            | "prod_i16"
            | "prod_i32"
            | "prod_i64"
            | "prod_u8"
            | "prod_u16"
            | "prod_u32"
            | "prod_u64"
            | "std_f8e4m3"
            | "std_f8e5m2"
            | "std_bf16"
            | "std_f16"
            | "std_f32"
            | "std_f64"
            | "var_f8e4m3"
            | "var_f8e5m2"
            | "var_bf16"
            | "var_f16"
            | "var_f32"
            | "var_f64"
            | "norm_f8e4m3"
            | "norm_f8e5m2"
            | "norm_bf16"
            | "norm_f16"
            | "norm_f32"
            | "norm_f64"
            | "argmax_bool"
            | "argmax_f8e4m3"
            | "argmax_f8e5m2"
            | "argmax_bf16"
            | "argmax_f16"
            | "argmax_f32"
            | "argmax_i8"
            | "argmax_i16"
            | "argmax_i32"
            | "argmax_i64"
            | "argmax_u8"
            | "argmax_u16"
            | "argmax_u32"
            | "argmax_u64"
            | "argmin_bool"
            | "argmin_f8e4m3"
            | "argmin_f8e5m2"
            | "argmin_bf16"
            | "argmin_f16"
            | "argmin_f32"
            | "argmin_i8"
            | "argmin_i16"
            | "argmin_i32"
            | "argmin_i64"
            | "argmin_u8"
            | "argmin_u16"
            | "argmin_u32"
            | "argmin_u64"
            | "any_bool"
            | "any_f8e4m3"
            | "any_f8e5m2"
            | "any_bf16"
            | "any_f16"
            | "any_f32"
            | "any_i8"
            | "any_i16"
            | "any_i32"
            | "any_i64"
            | "any_u8"
            | "any_u16"
            | "any_u32"
            | "any_u64"
            | "all_bool"
            | "all_f8e4m3"
            | "all_f8e5m2"
            | "all_bf16"
            | "all_f16"
            | "all_f32"
            | "all_i8"
            | "all_i16"
            | "all_i32"
            | "all_i64"
            | "all_u8"
            | "all_u16"
            | "all_u32"
            | "all_u64"
    ) {
        panic!("Unsupported reduce kernel: {}", name);
    }
}
