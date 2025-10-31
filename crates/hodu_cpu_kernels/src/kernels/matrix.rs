use crate::kernels::macros::ops;
use std::ffi::c_void;

// Define all matrix operations using the macro
ops!(matmul, dot);

/// Call matmul operation by kernel name - batched matrix multiplication with broadcasting
pub fn call_matmul(
    kernel_name: crate::kernels::macros::Kernel,
    lhs: *const c_void,
    rhs: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) {
    let num_els = metadata[0]; // First element should be num_els
    let num_dims = 0; // Not used for matrix ops, but kept for consistency

    unsafe {
        dispatch_matmul(kernel_name.0, lhs, rhs, output, num_els, num_dims, &metadata[1..]);
    }
}

/// Call dot operation by kernel name - tiled 2D matrix multiplication
pub fn call_dot(
    kernel_name: crate::kernels::macros::Kernel,
    lhs: *const c_void,
    rhs: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) {
    let m = metadata[0];
    let n = metadata[2];
    let num_els = m * n;
    let num_dims = 0; // Not used for matrix ops

    unsafe {
        dispatch_dot(kernel_name.0, lhs, rhs, output, num_els, num_dims, metadata.as_ptr());
    }
}

// Macro to automatically generate extern declarations and dispatch for matrix operations
macro_rules! declare_and_dispatch_matrix {
    ($($op:ident),* $(,)?) => {
        paste::paste! {
            // Extern C declarations for all operations and types
            extern "C" {
                $(
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

            // Dispatch function for matmul
            unsafe fn dispatch_matmul(
                name: &str,
                lhs: *const c_void,
                rhs: *const c_void,
                output: *mut c_void,
                num_els: usize,
                num_dims: usize,
                metadata: &[usize],
            ) {
                match name {
                    "matmul_f8e4m3" => matmul_f8e4m3(lhs, rhs, output, num_els, num_dims, metadata.as_ptr()),
                    "matmul_f8e5m2" => matmul_f8e5m2(lhs, rhs, output, num_els, num_dims, metadata.as_ptr()),
                    "matmul_bf16" => matmul_bf16(lhs, rhs, output, num_els, num_dims, metadata.as_ptr()),
                    "matmul_f16" => matmul_f16(lhs, rhs, output, num_els, num_dims, metadata.as_ptr()),
                    "matmul_f32" => matmul_f32(lhs, rhs, output, num_els, num_dims, metadata.as_ptr()),
                    "matmul_f64" => matmul_f64(lhs, rhs, output, num_els, num_dims, metadata.as_ptr()),
                    "matmul_u8" => matmul_u8(lhs, rhs, output, num_els, num_dims, metadata.as_ptr()),
                    "matmul_u16" => matmul_u16(lhs, rhs, output, num_els, num_dims, metadata.as_ptr()),
                    "matmul_u32" => matmul_u32(lhs, rhs, output, num_els, num_dims, metadata.as_ptr()),
                    "matmul_u64" => matmul_u64(lhs, rhs, output, num_els, num_dims, metadata.as_ptr()),
                    "matmul_i8" => matmul_i8(lhs, rhs, output, num_els, num_dims, metadata.as_ptr()),
                    "matmul_i16" => matmul_i16(lhs, rhs, output, num_els, num_dims, metadata.as_ptr()),
                    "matmul_i32" => matmul_i32(lhs, rhs, output, num_els, num_dims, metadata.as_ptr()),
                    "matmul_i64" => matmul_i64(lhs, rhs, output, num_els, num_dims, metadata.as_ptr()),
                    _ => panic!("Unsupported matmul kernel: {}", name),
                }
            }

            // Dispatch function for dot
            unsafe fn dispatch_dot(
                name: &str,
                lhs: *const c_void,
                rhs: *const c_void,
                output: *mut c_void,
                num_els: usize,
                num_dims: usize,
                metadata: *const usize,
            ) {
                match name {
                    "dot_f8e4m3" => dot_f8e4m3(lhs, rhs, output, num_els, num_dims, metadata),
                    "dot_f8e5m2" => dot_f8e5m2(lhs, rhs, output, num_els, num_dims, metadata),
                    "dot_bf16" => dot_bf16(lhs, rhs, output, num_els, num_dims, metadata),
                    "dot_f16" => dot_f16(lhs, rhs, output, num_els, num_dims, metadata),
                    "dot_f32" => dot_f32(lhs, rhs, output, num_els, num_dims, metadata),
                    "dot_f64" => dot_f64(lhs, rhs, output, num_els, num_dims, metadata),
                    "dot_u8" => dot_u8(lhs, rhs, output, num_els, num_dims, metadata),
                    "dot_u16" => dot_u16(lhs, rhs, output, num_els, num_dims, metadata),
                    "dot_u32" => dot_u32(lhs, rhs, output, num_els, num_dims, metadata),
                    "dot_u64" => dot_u64(lhs, rhs, output, num_els, num_dims, metadata),
                    "dot_i8" => dot_i8(lhs, rhs, output, num_els, num_dims, metadata),
                    "dot_i16" => dot_i16(lhs, rhs, output, num_els, num_dims, metadata),
                    "dot_i32" => dot_i32(lhs, rhs, output, num_els, num_dims, metadata),
                    "dot_i64" => dot_i64(lhs, rhs, output, num_els, num_dims, metadata),
                    _ => panic!("Unsupported dot kernel: {}", name),
                }
            }
        }
    };
}

// Declare matrix operations
declare_and_dispatch_matrix!(matmul, dot);
