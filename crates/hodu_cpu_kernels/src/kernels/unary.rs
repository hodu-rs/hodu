use crate::kernels::macros::ops;
use std::ffi::c_void;

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
    exp,
    exp2,
    exp10,
    ln,
    log2,
    log10,
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
pub fn call_unary(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) {
    let num_els: usize = shape.iter().product();
    let num_dims = shape.len();

    let mut metadata = Vec::with_capacity(num_dims * 2 + 1);
    metadata.extend_from_slice(shape);
    metadata.extend_from_slice(strides);
    metadata.push(offset);

    unsafe {
        dispatch_unary(kernel_name.0, input, output, num_els, num_dims, metadata.as_ptr());
    }
}

/// Call unary operation with scalar by kernel name
pub fn call_unary_scalar<T>(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    scalar_val: T,
) {
    let num_els: usize = shape.iter().product();
    let num_dims = shape.len();

    let mut metadata = Vec::with_capacity(num_dims * 2 + 1);
    metadata.extend_from_slice(shape);
    metadata.extend_from_slice(strides);
    metadata.push(offset);

    unsafe {
        dispatch_unary_scalar(
            kernel_name.0,
            input,
            output,
            num_els,
            num_dims,
            metadata.as_ptr(),
            &scalar_val as *const T as *const c_void,
        );
    }
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
                    fn [<$all_op _bool>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$all_op _f8e4m3>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$all_op _f8e5m2>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$all_op _bf16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$all_op _f16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$all_op _f32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$all_op _f64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$all_op _u8>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$all_op _u16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$all_op _u32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$all_op _u64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$all_op _i8>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$all_op _i16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$all_op _i32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$all_op _i64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                )*

                // Operations supporting signed types only (bool, float, signed int)
                $(
                    fn [<$signed_op _bool>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$signed_op _f8e4m3>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$signed_op _f8e5m2>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$signed_op _bf16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$signed_op _f16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$signed_op _f32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$signed_op _f64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$signed_op _i8>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$signed_op _i16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$signed_op _i32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$signed_op _i64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                )*

                // Operations supporting only float types
                $(
                    fn [<$float_op _bool>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$float_op _f8e4m3>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$float_op _f8e5m2>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$float_op _bf16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$float_op _f16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$float_op _f32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$float_op _f64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                )*

                // Scalar operations (all types, with scalar parameter)
                $(
                    fn [<$scalar_op _bool>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, scalar: *const c_void);
                    fn [<$scalar_op _f8e4m3>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, scalar: *const c_void);
                    fn [<$scalar_op _f8e5m2>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, scalar: *const c_void);
                    fn [<$scalar_op _bf16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, scalar: *const c_void);
                    fn [<$scalar_op _f16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, scalar: *const c_void);
                    fn [<$scalar_op _f32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, scalar: *const c_void);
                    fn [<$scalar_op _f64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, scalar: *const c_void);
                    fn [<$scalar_op _u8>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, scalar: *const c_void);
                    fn [<$scalar_op _u16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, scalar: *const c_void);
                    fn [<$scalar_op _u32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, scalar: *const c_void);
                    fn [<$scalar_op _u64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, scalar: *const c_void);
                    fn [<$scalar_op _i8>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, scalar: *const c_void);
                    fn [<$scalar_op _i16>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, scalar: *const c_void);
                    fn [<$scalar_op _i32>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, scalar: *const c_void);
                    fn [<$scalar_op _i64>](input: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize, scalar: *const c_void);
                )*
            }

            // Dispatch function
            unsafe fn dispatch_unary(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                num_els: usize,
                num_dims: usize,
                metadata: *const usize,
            ) {
                match name {
                    // Operations with all types
                    $(
                        concat!(stringify!($all_op), "_bool") => [<$all_op _bool>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($all_op), "_f8e4m3") => [<$all_op _f8e4m3>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($all_op), "_f8e5m2") => [<$all_op _f8e5m2>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($all_op), "_bf16") => [<$all_op _bf16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($all_op), "_f16") => [<$all_op _f16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($all_op), "_f32") => [<$all_op _f32>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($all_op), "_f64") => [<$all_op _f64>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($all_op), "_u8") => [<$all_op _u8>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($all_op), "_u16") => [<$all_op _u16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($all_op), "_u32") => [<$all_op _u32>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($all_op), "_u64") => [<$all_op _u64>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($all_op), "_i8") => [<$all_op _i8>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($all_op), "_i16") => [<$all_op _i16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($all_op), "_i32") => [<$all_op _i32>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($all_op), "_i64") => [<$all_op _i64>](input, output, num_els, num_dims, metadata),
                    )*

                    // Signed-only operations
                    $(
                        concat!(stringify!($signed_op), "_bool") => [<$signed_op _bool>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($signed_op), "_f8e4m3") => [<$signed_op _f8e4m3>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($signed_op), "_f8e5m2") => [<$signed_op _f8e5m2>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($signed_op), "_bf16") => [<$signed_op _bf16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($signed_op), "_f16") => [<$signed_op _f16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($signed_op), "_f32") => [<$signed_op _f32>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($signed_op), "_f64") => [<$signed_op _f64>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($signed_op), "_i8") => [<$signed_op _i8>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($signed_op), "_i16") => [<$signed_op _i16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($signed_op), "_i32") => [<$signed_op _i32>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($signed_op), "_i64") => [<$signed_op _i64>](input, output, num_els, num_dims, metadata),
                    )*

                    // Float-only operations
                    $(
                        concat!(stringify!($float_op), "_bool") => [<$float_op _bool>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($float_op), "_f8e4m3") => [<$float_op _f8e4m3>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($float_op), "_f8e5m2") => [<$float_op _f8e5m2>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($float_op), "_bf16") => [<$float_op _bf16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($float_op), "_f16") => [<$float_op _f16>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($float_op), "_f32") => [<$float_op _f32>](input, output, num_els, num_dims, metadata),
                        concat!(stringify!($float_op), "_f64") => [<$float_op _f64>](input, output, num_els, num_dims, metadata),
                    )*

                    _ => panic!("Unsupported unary kernel: {}", name),
                }
            }

            // Dispatch function for scalar operations
            unsafe fn dispatch_unary_scalar(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                num_els: usize,
                num_dims: usize,
                metadata: *const usize,
                scalar: *const c_void,
            ) {
                match name {
                    // Scalar operations (all types)
                    $(
                        concat!(stringify!($scalar_op), "_bool") => [<$scalar_op _bool>](input, output, num_els, num_dims, metadata, scalar),
                        concat!(stringify!($scalar_op), "_f8e4m3") => [<$scalar_op _f8e4m3>](input, output, num_els, num_dims, metadata, scalar),
                        concat!(stringify!($scalar_op), "_f8e5m2") => [<$scalar_op _f8e5m2>](input, output, num_els, num_dims, metadata, scalar),
                        concat!(stringify!($scalar_op), "_bf16") => [<$scalar_op _bf16>](input, output, num_els, num_dims, metadata, scalar),
                        concat!(stringify!($scalar_op), "_f16") => [<$scalar_op _f16>](input, output, num_els, num_dims, metadata, scalar),
                        concat!(stringify!($scalar_op), "_f32") => [<$scalar_op _f32>](input, output, num_els, num_dims, metadata, scalar),
                        concat!(stringify!($scalar_op), "_f64") => [<$scalar_op _f64>](input, output, num_els, num_dims, metadata, scalar),
                        concat!(stringify!($scalar_op), "_u8") => [<$scalar_op _u8>](input, output, num_els, num_dims, metadata, scalar),
                        concat!(stringify!($scalar_op), "_u16") => [<$scalar_op _u16>](input, output, num_els, num_dims, metadata, scalar),
                        concat!(stringify!($scalar_op), "_u32") => [<$scalar_op _u32>](input, output, num_els, num_dims, metadata, scalar),
                        concat!(stringify!($scalar_op), "_u64") => [<$scalar_op _u64>](input, output, num_els, num_dims, metadata, scalar),
                        concat!(stringify!($scalar_op), "_i8") => [<$scalar_op _i8>](input, output, num_els, num_dims, metadata, scalar),
                        concat!(stringify!($scalar_op), "_i16") => [<$scalar_op _i16>](input, output, num_els, num_dims, metadata, scalar),
                        concat!(stringify!($scalar_op), "_i32") => [<$scalar_op _i32>](input, output, num_els, num_dims, metadata, scalar),
                        concat!(stringify!($scalar_op), "_i64") => [<$scalar_op _i64>](input, output, num_els, num_dims, metadata, scalar),
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
        exp, exp2, exp10, ln, log2, log10
    ],
    scalar_ops: [
        add_scalar, sub_scalar, mul_scalar, div_scalar, pow_scalar,
        maximum_scalar, minimum_scalar,
        eq_scalar, ne_scalar, lt_scalar, le_scalar, gt_scalar, ge_scalar
    ]
);
