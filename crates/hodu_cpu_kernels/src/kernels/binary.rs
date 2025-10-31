use crate::{error::Result, kernels::macros::ops};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::ffi::c_void;

// Define all binary operations using the macro
ops!(
    add,
    sub,
    mul,
    div,
    pow,
    maximum,
    minimum,
    logical_and,
    logical_or,
    logical_xor,
    eq,
    ne,
    lt,
    le,
    gt,
    ge
);

/// Call binary operation by kernel name
#[allow(clippy::too_many_arguments)]
pub fn call_binary(
    kernel_name: crate::kernels::macros::Kernel,
    lhs: *const c_void,
    rhs: *const c_void,
    output: *mut c_void,
    shape: &[usize],
    lhs_strides: &[usize],
    rhs_strides: &[usize],
    lhs_offset: usize,
    rhs_offset: usize,
) -> Result<()> {
    let num_els: usize = shape.iter().product();
    let num_dims = shape.len();

    let mut metadata = Vec::with_capacity(num_dims * 3 + 2);
    metadata.extend_from_slice(shape);
    metadata.extend_from_slice(lhs_strides);
    metadata.extend_from_slice(rhs_strides);
    metadata.push(lhs_offset);
    metadata.push(rhs_offset);

    unsafe {
        dispatch_binary(kernel_name.0, lhs, rhs, output, num_els, num_dims, metadata.as_ptr());
    }

    Ok(())
}

// Macro to automatically generate extern declarations and dispatch for all binary operations and types
macro_rules! declare_and_dispatch_binary {
    ($($op:ident),* $(,)?) => {
        paste::paste! {
            // Extern C declarations for all operations and types
            extern "C" {
                $(
                    fn [<$op _bool>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f8e4m3>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f8e5m2>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _bf16>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f16>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f32>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f64>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u8>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u16>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u32>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u64>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i8>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i16>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i32>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i64>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                )*
            }

            // Dispatch function
            unsafe fn dispatch_binary(
                name: &str,
                lhs: *const c_void,
                rhs: *const c_void,
                output: *mut c_void,
                num_els: usize,
                num_dims: usize,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!(stringify!($op), "_bool") => [<$op _bool>](lhs, rhs, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f8e4m3") => [<$op _f8e4m3>](lhs, rhs, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f8e5m2") => [<$op _f8e5m2>](lhs, rhs, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_bf16") => [<$op _bf16>](lhs, rhs, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f16") => [<$op _f16>](lhs, rhs, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f32") => [<$op _f32>](lhs, rhs, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f64") => [<$op _f64>](lhs, rhs, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u8") => [<$op _u8>](lhs, rhs, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u16") => [<$op _u16>](lhs, rhs, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u32") => [<$op _u32>](lhs, rhs, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u64") => [<$op _u64>](lhs, rhs, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i8") => [<$op _i8>](lhs, rhs, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i16") => [<$op _i16>](lhs, rhs, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i32") => [<$op _i32>](lhs, rhs, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i64") => [<$op _i64>](lhs, rhs, output, num_els, num_dims, metadata),
                    )*
                    _ => panic!("Unsupported binary kernel: {}", name),
                }
            }
        }
    };
}

// Declare all binary operations
declare_and_dispatch_binary!(
    add,
    sub,
    mul,
    div,
    pow,
    maximum,
    minimum,
    logical_and,
    logical_or,
    logical_xor,
    eq,
    ne,
    lt,
    le,
    gt,
    ge
);
