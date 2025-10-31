use crate::{error::Result, kernels::macros::ops};
use core::ffi::c_void;

// Define all reduce operations using the macro
ops!(
    reduce_sum,
    reduce_mean,
    reduce_max,
    reduce_min,
    reduce_prod,
    reduce_std,
    reduce_var,
    reduce_norm,
    reduce_argmax,
    reduce_argmin,
    reduce_any,
    reduce_all
);

/// Call reduce operation by kernel name
pub fn call_reduce(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    // metadata layout: [shape_len, shape..., strides..., offset, output_shape_len, output_shape...,
    //                   num_reduce_dims, reduce_dims..., keep_dim, reduce_size]
    let shape_len = metadata[0];
    let offset_idx = 1 + shape_len * 2; // After shape and strides
    let output_shape_len_idx = offset_idx + 1;
    let output_shape_len = metadata[output_shape_len_idx];

    // Calculate num_els (output elements)
    let output_shape_start = output_shape_len_idx + 1;
    let output_shape_end = output_shape_start + output_shape_len;
    let num_els: usize = metadata[output_shape_start..output_shape_end].iter().product();

    // Get reduce_size from the end of metadata
    let reduce_size = metadata[metadata.len() - 1];

    let num_dims = shape_len;

    // Prepare metadata for C function (skip shape_len)
    let c_metadata = &metadata[1..];

    unsafe {
        dispatch_reduce(
            kernel_name.0,
            input,
            output,
            num_els,
            num_dims,
            c_metadata.as_ptr(),
            reduce_size,
        );
    }

    Ok(())
}

// Macro to declare extern C functions and dispatch for reduce operations
macro_rules! declare_reduce_ops {
    // All types (sum, max, min, prod)
    (all_types: $($op:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<$op _f8e4m3>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _f8e5m2>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _bf16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _f16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _f32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _f64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _i8>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _i16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _i32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _i64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _u8>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _u16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _u32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _u64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                )*
            }
        }
    };

    // Float types only (mean, norm, std, var)
    (float_types: $($op:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<$op _f8e4m3>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _f8e5m2>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _bf16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _f16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _f32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _f64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                )*
            }
        }
    };

    // All and bool types
    (all_and_bool_types: $($op:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<$op _bool>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _f8e4m3>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _f8e5m2>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _bf16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _f16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _f32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _f64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _i8>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _i16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _i32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _i64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _u8>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _u16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _u32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                    fn [<$op _u64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, reduce_size: usize);
                )*
            }
        }
    };
}

// Declare all reduce operations
declare_reduce_ops!(all_types: reduce_sum, reduce_max, reduce_min, reduce_prod);
declare_reduce_ops!(float_types: reduce_std, reduce_var, reduce_mean, reduce_norm);
declare_reduce_ops!(all_and_bool_types: reduce_argmax, reduce_argmin, reduce_any, reduce_all);

// Macro to generate dispatch match arms
macro_rules! dispatch_all_types {
    ($name:expr, $input:expr, $output:expr, $num_els:expr, $num_dims:expr, $metadata:expr, $reduce_size:expr, $($op:ident),* $(,)?) => {
        paste::paste! {
            match $name {
                $(
                    concat!(stringify!($op), "_f8e4m3") => [<$op _f8e4m3>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_f8e5m2") => [<$op _f8e5m2>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_bf16") => [<$op _bf16>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_f16") => [<$op _f16>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_f32") => [<$op _f32>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_f64") => [<$op _f64>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_i8") => [<$op _i8>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_i16") => [<$op _i16>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_i32") => [<$op _i32>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_i64") => [<$op _i64>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_u8") => [<$op _u8>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_u16") => [<$op _u16>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_u32") => [<$op _u32>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_u64") => [<$op _u64>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                )*
                _ => {}
            }
        }
    };
}

macro_rules! dispatch_float_types {
    ($name:expr, $input:expr, $output:expr, $num_els:expr, $num_dims:expr, $metadata:expr, $reduce_size:expr, $($op:ident),* $(,)?) => {
        paste::paste! {
            match $name {
                $(
                    concat!(stringify!($op), "_f8e4m3") => [<$op _f8e4m3>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_f8e5m2") => [<$op _f8e5m2>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_bf16") => [<$op _bf16>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_f16") => [<$op _f16>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_f32") => [<$op _f32>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_f64") => [<$op _f64>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                )*
                _ => {}
            }
        }
    };
}

macro_rules! dispatch_all_and_bool_types {
    ($name:expr, $input:expr, $output:expr, $num_els:expr, $num_dims:expr, $metadata:expr, $reduce_size:expr, $($op:ident),* $(,)?) => {
        paste::paste! {
            match $name {
                $(
                    concat!(stringify!($op), "_bool") => [<$op _bool>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_f8e4m3") => [<$op _f8e4m3>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_f8e5m2") => [<$op _f8e5m2>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_bf16") => [<$op _bf16>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_f16") => [<$op _f16>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_f32") => [<$op _f32>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_f64") => [<$op _f64>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_i8") => [<$op _i8>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_i16") => [<$op _i16>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_i32") => [<$op _i32>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_i64") => [<$op _i64>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_u8") => [<$op _u8>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_u16") => [<$op _u16>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_u32") => [<$op _u32>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                    concat!(stringify!($op), "_u64") => [<$op _u64>]($input, $output, $num_els, $num_dims, $metadata, $reduce_size),
                )*
                _ => {}
            }
        }
    };
}

// Dispatch function for reduce operations
unsafe fn dispatch_reduce(
    name: &str,
    input: *const c_void,
    output: *mut c_void,
    num_els: usize,
    num_dims: usize,
    metadata: *const usize,
    reduce_size: usize,
) {
    dispatch_all_types!(
        name,
        input,
        output,
        num_els,
        num_dims,
        metadata,
        reduce_size,
        reduce_sum,
        reduce_max,
        reduce_min,
        reduce_prod
    );

    dispatch_float_types!(
        name,
        input,
        output,
        num_els,
        num_dims,
        metadata,
        reduce_size,
        reduce_std,
        reduce_var,
        reduce_mean,
        reduce_norm
    );

    dispatch_all_and_bool_types!(
        name,
        input,
        output,
        num_els,
        num_dims,
        metadata,
        reduce_size,
        reduce_argmax,
        reduce_argmin,
        reduce_any,
        reduce_all
    );

    // If we get here, kernel was not found
    if !matches!(
        name,
        "reduce_sum_f8e4m3"
            | "reduce_sum_f8e5m2"
            | "reduce_sum_bf16"
            | "reduce_sum_f16"
            | "reduce_sum_f32"
            | "reduce_sum_f64"
            | "reduce_sum_i8"
            | "reduce_sum_i16"
            | "reduce_sum_i32"
            | "reduce_sum_i64"
            | "reduce_sum_u8"
            | "reduce_sum_u16"
            | "reduce_sum_u32"
            | "reduce_sum_u64"
            | "reduce_mean_f8e4m3"
            | "reduce_mean_f8e5m2"
            | "reduce_mean_bf16"
            | "reduce_mean_f16"
            | "reduce_mean_f32"
            | "reduce_mean_f64"
            | "reduce_max_f8e4m3"
            | "reduce_max_f8e5m2"
            | "reduce_max_bf16"
            | "reduce_max_f16"
            | "reduce_max_f32"
            | "reduce_max_f64"
            | "reduce_max_i8"
            | "reduce_max_i16"
            | "reduce_max_i32"
            | "reduce_max_i64"
            | "reduce_max_u8"
            | "reduce_max_u16"
            | "reduce_max_u32"
            | "reduce_max_u64"
            | "reduce_min_f8e4m3"
            | "reduce_min_f8e5m2"
            | "reduce_min_bf16"
            | "reduce_min_f16"
            | "reduce_min_f32"
            | "reduce_min_f64"
            | "reduce_min_i8"
            | "reduce_min_i16"
            | "reduce_min_i32"
            | "reduce_min_i64"
            | "reduce_min_u8"
            | "reduce_min_u16"
            | "reduce_min_u32"
            | "reduce_min_u64"
            | "reduce_prod_f8e4m3"
            | "reduce_prod_f8e5m2"
            | "reduce_prod_bf16"
            | "reduce_prod_f16"
            | "reduce_prod_f32"
            | "reduce_prod_f64"
            | "reduce_prod_i8"
            | "reduce_prod_i16"
            | "reduce_prod_i32"
            | "reduce_prod_i64"
            | "reduce_prod_u8"
            | "reduce_prod_u16"
            | "reduce_prod_u32"
            | "reduce_prod_u64"
            | "reduce_std_f8e4m3"
            | "reduce_std_f8e5m2"
            | "reduce_std_bf16"
            | "reduce_std_f16"
            | "reduce_std_f32"
            | "reduce_std_f64"
            | "reduce_var_f8e4m3"
            | "reduce_var_f8e5m2"
            | "reduce_var_bf16"
            | "reduce_var_f16"
            | "reduce_var_f32"
            | "reduce_var_f64"
            | "reduce_norm_f8e4m3"
            | "reduce_norm_f8e5m2"
            | "reduce_norm_bf16"
            | "reduce_norm_f16"
            | "reduce_norm_f32"
            | "reduce_norm_f64"
            | "reduce_argmax_bool"
            | "reduce_argmax_f8e4m3"
            | "reduce_argmax_f8e5m2"
            | "reduce_argmax_bf16"
            | "reduce_argmax_f16"
            | "reduce_argmax_f32"
            | "reduce_argmax_i8"
            | "reduce_argmax_i16"
            | "reduce_argmax_i32"
            | "reduce_argmax_i64"
            | "reduce_argmax_u8"
            | "reduce_argmax_u16"
            | "reduce_argmax_u32"
            | "reduce_argmax_u64"
            | "reduce_argmin_bool"
            | "reduce_argmin_f8e4m3"
            | "reduce_argmin_f8e5m2"
            | "reduce_argmin_bf16"
            | "reduce_argmin_f16"
            | "reduce_argmin_f32"
            | "reduce_argmin_i8"
            | "reduce_argmin_i16"
            | "reduce_argmin_i32"
            | "reduce_argmin_i64"
            | "reduce_argmin_u8"
            | "reduce_argmin_u16"
            | "reduce_argmin_u32"
            | "reduce_argmin_u64"
            | "reduce_any_bool"
            | "reduce_any_f8e4m3"
            | "reduce_any_f8e5m2"
            | "reduce_any_bf16"
            | "reduce_any_f16"
            | "reduce_any_f32"
            | "reduce_any_i8"
            | "reduce_any_i16"
            | "reduce_any_i32"
            | "reduce_any_i64"
            | "reduce_any_u8"
            | "reduce_any_u16"
            | "reduce_any_u32"
            | "reduce_any_u64"
            | "reduce_all_bool"
            | "reduce_all_f8e4m3"
            | "reduce_all_f8e5m2"
            | "reduce_all_bf16"
            | "reduce_all_f16"
            | "reduce_all_f32"
            | "reduce_all_i8"
            | "reduce_all_i16"
            | "reduce_all_i32"
            | "reduce_all_i64"
            | "reduce_all_u8"
            | "reduce_all_u16"
            | "reduce_all_u32"
            | "reduce_all_u64"
    ) {
        panic!("Unsupported reduce kernel: {}", name);
    }
}
