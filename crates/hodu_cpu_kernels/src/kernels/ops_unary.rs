//! Unary tensor operations
//!
//! This module provides element-wise unary operations on tensors, including:
//! - Basic arithmetic: neg, abs, sign, square, sqrt, recip
//! - Activation functions: relu, sigmoid, tanh, gelu, softplus, silu, mish
//! - Trigonometric: sin, cos, tan
//! - Exponential/logarithmic: exp, exp2, exp10, ln, log2, log10
//! - Logical: logical_not
//! - Scalar operations: arithmetic and comparison operations with a scalar value
//!
//! All operations support strided tensor access and multiple data types.

use crate::{error::Result, kernels::macros::ops};
use core::ffi::c_void;

// Define all unary operations using the macro
ops!(
    neg,
    abs,
    sign,
    square,
    sqrt,
    recip,
    relu,
    sigmoid,
    tanh,
    gelu,
    softplus,
    silu,
    mish,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    atanh,
    exp,
    exp2,
    exp10,
    ln,
    log2,
    log10,
    ceil,
    floor,
    round,
    logical_not,
    add_scalar,
    sub_scalar,
    mul_scalar,
    div_scalar,
    pow_scalar,
    maximum_scalar,
    minimum_scalar,
    eq_scalar,
    ne_scalar,
    lt_scalar,
    le_scalar,
    gt_scalar,
    ge_scalar
);

/// Call unary operation by kernel name
///
/// Applies an element-wise unary operation to the input tensor.
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape
/// - metadata[2+num_dims..2+2*num_dims]: strides
/// - metadata[2+2*num_dims]: offset
///
/// # Safety
/// - `input` must point to valid tensor data of the appropriate type
/// - `output` must point to a valid output buffer with sufficient capacity
/// - Metadata must accurately describe the tensor layout
pub fn call_ops_unary(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_unary(kernel_name.0, input, output, metadata.as_ptr());
    }

    Ok(())
}

/// Call unary operation with scalar by kernel name
///
/// Applies an element-wise operation combining each tensor element with a scalar value.
/// Used for operations like add_scalar, mul_scalar, and comparison operations.
///
/// # Metadata layout (same as call_ops_unary)
/// - metadata[0]: num_els (total number of elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape
/// - metadata[2+num_dims..2+2*num_dims]: strides
/// - metadata[2+2*num_dims]: offset
///
/// # Safety
/// - `input` must point to valid tensor data of the appropriate type
/// - `output` must point to a valid output buffer with sufficient capacity
/// - `scalar_val` must be of the same type as the tensor elements
/// - Metadata must accurately describe the tensor layout
pub fn call_ops_unary_scalar<T>(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
    scalar_val: T,
) -> Result<()> {
    unsafe {
        dispatch_unary_scalar(
            kernel_name.0,
            input,
            output,
            metadata.as_ptr(),
            &scalar_val as *const T as *const c_void,
        );
    }

    Ok(())
}

// Macro to automatically generate extern declarations and dispatch for all operations and types
macro_rules! declare_and_dispatch_unary {
    (
        all_types: [$($all_op:ident),* $(,)?],
        signed_types: [$($signed_op:ident),* $(,)?],
        float_types: [$($float_op:ident),* $(,)?],
        scalar_ops: [$($scalar_op:ident),* $(,)?]
    ) => {
        paste::paste! {
            // Extern C declarations
            extern "C" {
                // Operations supporting all types (including unsigned)
                $(
                    fn [<hodu_cpu_ $all_op _bool>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $all_op _f8e4m3>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $all_op _f8e5m2>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $all_op _bf16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $all_op _f16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $all_op _f32>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $all_op _f64>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $all_op _u8>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $all_op _u16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $all_op _u32>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $all_op _u64>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $all_op _i8>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $all_op _i16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $all_op _i32>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $all_op _i64>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                )*

                // Operations supporting signed types only (bool, float, signed int)
                $(
                    fn [<hodu_cpu_ $signed_op _bool>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $signed_op _f8e4m3>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $signed_op _f8e5m2>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $signed_op _bf16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $signed_op _f16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $signed_op _f32>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $signed_op _f64>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $signed_op _i8>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $signed_op _i16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $signed_op _i32>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $signed_op _i64>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                )*

                // Operations supporting only float types
                $(
                    fn [<hodu_cpu_ $float_op _bool>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $float_op _f8e4m3>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $float_op _f8e5m2>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $float_op _bf16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $float_op _f16>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $float_op _f32>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $float_op _f64>](input: *const c_void, output: *mut c_void, metadata: *const usize);
                )*

                // Scalar operations (all types, with scalar parameter)
                $(
                    fn [<hodu_cpu_ $scalar_op _bool>](input: *const c_void, output: *mut c_void, metadata: *const usize, scalar: *const c_void);
                    fn [<hodu_cpu_ $scalar_op _f8e4m3>](input: *const c_void, output: *mut c_void, metadata: *const usize, scalar: *const c_void);
                    fn [<hodu_cpu_ $scalar_op _f8e5m2>](input: *const c_void, output: *mut c_void, metadata: *const usize, scalar: *const c_void);
                    fn [<hodu_cpu_ $scalar_op _bf16>](input: *const c_void, output: *mut c_void, metadata: *const usize, scalar: *const c_void);
                    fn [<hodu_cpu_ $scalar_op _f16>](input: *const c_void, output: *mut c_void, metadata: *const usize, scalar: *const c_void);
                    fn [<hodu_cpu_ $scalar_op _f32>](input: *const c_void, output: *mut c_void, metadata: *const usize, scalar: *const c_void);
                    fn [<hodu_cpu_ $scalar_op _f64>](input: *const c_void, output: *mut c_void, metadata: *const usize, scalar: *const c_void);
                    fn [<hodu_cpu_ $scalar_op _u8>](input: *const c_void, output: *mut c_void, metadata: *const usize, scalar: *const c_void);
                    fn [<hodu_cpu_ $scalar_op _u16>](input: *const c_void, output: *mut c_void, metadata: *const usize, scalar: *const c_void);
                    fn [<hodu_cpu_ $scalar_op _u32>](input: *const c_void, output: *mut c_void, metadata: *const usize, scalar: *const c_void);
                    fn [<hodu_cpu_ $scalar_op _u64>](input: *const c_void, output: *mut c_void, metadata: *const usize, scalar: *const c_void);
                    fn [<hodu_cpu_ $scalar_op _i8>](input: *const c_void, output: *mut c_void, metadata: *const usize, scalar: *const c_void);
                    fn [<hodu_cpu_ $scalar_op _i16>](input: *const c_void, output: *mut c_void, metadata: *const usize, scalar: *const c_void);
                    fn [<hodu_cpu_ $scalar_op _i32>](input: *const c_void, output: *mut c_void, metadata: *const usize, scalar: *const c_void);
                    fn [<hodu_cpu_ $scalar_op _i64>](input: *const c_void, output: *mut c_void, metadata: *const usize, scalar: *const c_void);
                )*
            }

            // Dispatch function
            unsafe fn dispatch_unary(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    // Operations with all types
                    $(
                        concat!("hodu_cpu_", stringify!($all_op), "_bool") => [<hodu_cpu_ $all_op _bool>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($all_op), "_f8e4m3") => [<hodu_cpu_ $all_op _f8e4m3>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($all_op), "_f8e5m2") => [<hodu_cpu_ $all_op _f8e5m2>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($all_op), "_bf16") => [<hodu_cpu_ $all_op _bf16>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($all_op), "_f16") => [<hodu_cpu_ $all_op _f16>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($all_op), "_f32") => [<hodu_cpu_ $all_op _f32>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($all_op), "_f64") => [<hodu_cpu_ $all_op _f64>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($all_op), "_u8") => [<hodu_cpu_ $all_op _u8>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($all_op), "_u16") => [<hodu_cpu_ $all_op _u16>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($all_op), "_u32") => [<hodu_cpu_ $all_op _u32>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($all_op), "_u64") => [<hodu_cpu_ $all_op _u64>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($all_op), "_i8") => [<hodu_cpu_ $all_op _i8>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($all_op), "_i16") => [<hodu_cpu_ $all_op _i16>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($all_op), "_i32") => [<hodu_cpu_ $all_op _i32>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($all_op), "_i64") => [<hodu_cpu_ $all_op _i64>](input, output, metadata),
                    )*

                    // Signed-only operations
                    $(
                        concat!("hodu_cpu_", stringify!($signed_op), "_bool") => [<hodu_cpu_ $signed_op _bool>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($signed_op), "_f8e4m3") => [<hodu_cpu_ $signed_op _f8e4m3>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($signed_op), "_f8e5m2") => [<hodu_cpu_ $signed_op _f8e5m2>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($signed_op), "_bf16") => [<hodu_cpu_ $signed_op _bf16>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($signed_op), "_f16") => [<hodu_cpu_ $signed_op _f16>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($signed_op), "_f32") => [<hodu_cpu_ $signed_op _f32>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($signed_op), "_f64") => [<hodu_cpu_ $signed_op _f64>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($signed_op), "_i8") => [<hodu_cpu_ $signed_op _i8>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($signed_op), "_i16") => [<hodu_cpu_ $signed_op _i16>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($signed_op), "_i32") => [<hodu_cpu_ $signed_op _i32>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($signed_op), "_i64") => [<hodu_cpu_ $signed_op _i64>](input, output, metadata),
                    )*

                    // Float-only operations
                    $(
                        concat!("hodu_cpu_", stringify!($float_op), "_bool") => [<hodu_cpu_ $float_op _bool>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($float_op), "_f8e4m3") => [<hodu_cpu_ $float_op _f8e4m3>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($float_op), "_f8e5m2") => [<hodu_cpu_ $float_op _f8e5m2>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($float_op), "_bf16") => [<hodu_cpu_ $float_op _bf16>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($float_op), "_f16") => [<hodu_cpu_ $float_op _f16>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($float_op), "_f32") => [<hodu_cpu_ $float_op _f32>](input, output, metadata),
                        concat!("hodu_cpu_", stringify!($float_op), "_f64") => [<hodu_cpu_ $float_op _f64>](input, output, metadata),
                    )*

                    _ => panic!("Unsupported unary kernel: {}", name),
                }
            }

            // Dispatch function for scalar operations
            unsafe fn dispatch_unary_scalar(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
                scalar: *const c_void,
            ) {
                match name {
                    // Scalar operations (all types)
                    $(
                        concat!("hodu_cpu_", stringify!($scalar_op), "_bool") => [<hodu_cpu_ $scalar_op _bool>](input, output, metadata, scalar),
                        concat!("hodu_cpu_", stringify!($scalar_op), "_f8e4m3") => [<hodu_cpu_ $scalar_op _f8e4m3>](input, output, metadata, scalar),
                        concat!("hodu_cpu_", stringify!($scalar_op), "_f8e5m2") => [<hodu_cpu_ $scalar_op _f8e5m2>](input, output, metadata, scalar),
                        concat!("hodu_cpu_", stringify!($scalar_op), "_bf16") => [<hodu_cpu_ $scalar_op _bf16>](input, output, metadata, scalar),
                        concat!("hodu_cpu_", stringify!($scalar_op), "_f16") => [<hodu_cpu_ $scalar_op _f16>](input, output, metadata, scalar),
                        concat!("hodu_cpu_", stringify!($scalar_op), "_f32") => [<hodu_cpu_ $scalar_op _f32>](input, output, metadata, scalar),
                        concat!("hodu_cpu_", stringify!($scalar_op), "_f64") => [<hodu_cpu_ $scalar_op _f64>](input, output, metadata, scalar),
                        concat!("hodu_cpu_", stringify!($scalar_op), "_u8") => [<hodu_cpu_ $scalar_op _u8>](input, output, metadata, scalar),
                        concat!("hodu_cpu_", stringify!($scalar_op), "_u16") => [<hodu_cpu_ $scalar_op _u16>](input, output, metadata, scalar),
                        concat!("hodu_cpu_", stringify!($scalar_op), "_u32") => [<hodu_cpu_ $scalar_op _u32>](input, output, metadata, scalar),
                        concat!("hodu_cpu_", stringify!($scalar_op), "_u64") => [<hodu_cpu_ $scalar_op _u64>](input, output, metadata, scalar),
                        concat!("hodu_cpu_", stringify!($scalar_op), "_i8") => [<hodu_cpu_ $scalar_op _i8>](input, output, metadata, scalar),
                        concat!("hodu_cpu_", stringify!($scalar_op), "_i16") => [<hodu_cpu_ $scalar_op _i16>](input, output, metadata, scalar),
                        concat!("hodu_cpu_", stringify!($scalar_op), "_i32") => [<hodu_cpu_ $scalar_op _i32>](input, output, metadata, scalar),
                        concat!("hodu_cpu_", stringify!($scalar_op), "_i64") => [<hodu_cpu_ $scalar_op _i64>](input, output, metadata, scalar),
                    )*
                    _ => panic!("Unsupported unary scalar kernel: {}", name),
                }
            }
        }
    };
}

// Declare all operations
declare_and_dispatch_unary!(
    all_types: [
        abs, sign, square, sqrt, recip, logical_not
    ],
    signed_types: [
        neg
    ],
    float_types: [
        relu, sigmoid, tanh, gelu, softplus, silu, mish,
        sin, cos, tan,
        exp, exp2, exp10, ln, log2, log10,
        ceil, floor, round
    ],
    scalar_ops: [
        add_scalar, sub_scalar, mul_scalar, div_scalar, pow_scalar,
        maximum_scalar, minimum_scalar,
        eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar
    ]
);
