//! Matrix operations for linear algebra computations
//!
//! This module provides:
//! - `matmul`: Batched matrix multiplication with broadcasting support
//! - `dot`: Optimized 2D matrix multiplication
//!
//! Both operations support various numeric types including floating point and integers.

use crate::{error::Result, kernels::macros::ops};
use core::ffi::c_void;

// Define all matrix operations using the macro
ops!(matmul, dot);

/// Execute a batched matrix multiplication operation with broadcasting
///
/// Performs batched matrix multiplication (C = A @ B) with support for broadcasting
/// across batch dimensions. Supports arbitrary batch dimensions for both operands.
///
/// # Arguments
/// * `kernel_name` - The matmul kernel to execute (e.g., matmul::F32)
/// * `lhs` - Pointer to left-hand side tensor (A)
/// * `rhs` - Pointer to right-hand side tensor (B)
/// * `output` - Pointer to output tensor buffer (C)
/// * `metadata` - Tensor metadata array (see layout below)
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: lhs_ndim (number of dimensions in lhs)
/// - metadata[2]: rhs_ndim (number of dimensions in rhs)
/// - metadata[3]: batch_ndim (number of batch dimensions in output)
/// - metadata[4..4+lhs_ndim]: lhs_shape (shape of lhs tensor)
/// - metadata[4+lhs_ndim..4+lhs_ndim+rhs_ndim]: rhs_shape (shape of rhs tensor)
/// - metadata[4+lhs_ndim+rhs_ndim..4+lhs_ndim+rhs_ndim+batch_ndim]: batch_shape (broadcasted batch shape)
/// - metadata[4+lhs_ndim+rhs_ndim+batch_ndim..4+2*lhs_ndim+rhs_ndim+batch_ndim]: lhs_strides
/// - metadata[4+2*lhs_ndim+rhs_ndim+batch_ndim..4+2*lhs_ndim+2*rhs_ndim+batch_ndim]: rhs_strides
/// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim]: lhs_offset (starting offset in lhs)
/// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+1]: rhs_offset (starting offset in rhs)
/// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+2]: M (rows of lhs matrix)
/// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+3]: K (cols of lhs / rows of rhs)
/// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+4]: N (cols of rhs matrix)
///
/// # Broadcasting
/// Batch dimensions are broadcasted automatically. Dimensions of size 1 are
/// broadcasted to match the output batch shape.
///
/// # Safety
/// This function uses unsafe FFI calls to C kernels. Caller must ensure:
/// - All pointers are valid and properly aligned
/// - Metadata accurately describes tensor layout
/// - Output buffer has sufficient capacity
/// - Matrix dimensions are compatible: lhs[..., M, K] @ rhs[..., K, N] = output[..., M, N]
///
/// # Returns
/// Returns `Ok(())` on success.
pub fn call_ops_matmul(
    kernel_name: crate::kernels::macros::Kernel,
    lhs: *const c_void,
    rhs: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_matmul(kernel_name.0, lhs, rhs, output, metadata);
    }

    Ok(())
}

/// Execute a 2D matrix multiplication operation
///
/// Performs optimized 2D matrix multiplication (C = A @ B) for two 2D matrices only.
/// This is a simplified version of matmul without batch dimensions.
///
/// # Arguments
/// * `kernel_name` - The dot kernel to execute (e.g., dot::F32)
/// * `lhs` - Pointer to left-hand side matrix (A)
/// * `rhs` - Pointer to right-hand side matrix (B)
/// * `output` - Pointer to output matrix buffer (C)
/// * `metadata` - Tensor metadata array (see layout below)
///
/// # Metadata layout
/// - metadata[0]: M (number of rows in lhs)
/// - metadata[1]: K (number of cols in lhs / rows in rhs)
/// - metadata[2]: N (number of cols in rhs)
/// - metadata[3]: lhs_stride_m (stride for lhs rows)
/// - metadata[4]: lhs_stride_k (stride for lhs cols)
/// - metadata[5]: rhs_stride_k (stride for rhs rows)
/// - metadata[6]: rhs_stride_n (stride for rhs cols)
/// - metadata[7]: lhs_offset (starting offset in lhs)
/// - metadata[8]: rhs_offset (starting offset in rhs)
///
/// # Safety
/// This function uses unsafe FFI calls to C kernels. Caller must ensure:
/// - All pointers are valid and properly aligned
/// - Metadata accurately describes tensor layout
/// - Output buffer has sufficient capacity (M * N elements)
/// - Matrix dimensions are compatible: lhs[M, K] @ rhs[K, N] = output[M, N]
///
/// # Returns
/// Returns `Ok(())` on success.
pub fn call_ops_dot(
    kernel_name: crate::kernels::macros::Kernel,
    lhs: *const c_void,
    rhs: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_dot(kernel_name.0, lhs, rhs, output, metadata.as_ptr());
    }

    Ok(())
}

/// Macro to generate extern C declarations and dispatch logic for matrix operations
///
/// This macro generates FFI bindings for all supported numeric types.
macro_rules! declare_and_dispatch_matrix {
    ($($op:ident),* $(,)?) => {
        paste::paste! {
            // Extern C declarations for all operations and types
            extern "C" {
                $(
                    fn [<hodu_cpu_ $op _f8e4m3>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f8e5m2>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _bf16>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f16>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f32>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f64>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u8>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u16>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u32>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u64>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i8>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i16>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i32>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i64>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
                )*
            }

            // Dispatch function for matmul
            unsafe fn dispatch_matmul(
                name: &str,
                lhs: *const c_void,
                rhs: *const c_void,
                output: *mut c_void,
                metadata: &[usize],
            ) {
                match name {
                    "hodu_cpu_matmul_f8e4m3" => hodu_cpu_matmul_f8e4m3(lhs, rhs, output, metadata.as_ptr()),
                    "hodu_cpu_matmul_f8e5m2" => hodu_cpu_matmul_f8e5m2(lhs, rhs, output, metadata.as_ptr()),
                    "hodu_cpu_matmul_bf16" => hodu_cpu_matmul_bf16(lhs, rhs, output, metadata.as_ptr()),
                    "hodu_cpu_matmul_f16" => hodu_cpu_matmul_f16(lhs, rhs, output, metadata.as_ptr()),
                    "hodu_cpu_matmul_f32" => hodu_cpu_matmul_f32(lhs, rhs, output, metadata.as_ptr()),
                    "hodu_cpu_matmul_f64" => hodu_cpu_matmul_f64(lhs, rhs, output, metadata.as_ptr()),
                    "hodu_cpu_matmul_u8" => hodu_cpu_matmul_u8(lhs, rhs, output, metadata.as_ptr()),
                    "hodu_cpu_matmul_u16" => hodu_cpu_matmul_u16(lhs, rhs, output, metadata.as_ptr()),
                    "hodu_cpu_matmul_u32" => hodu_cpu_matmul_u32(lhs, rhs, output, metadata.as_ptr()),
                    "hodu_cpu_matmul_u64" => hodu_cpu_matmul_u64(lhs, rhs, output, metadata.as_ptr()),
                    "hodu_cpu_matmul_i8" => hodu_cpu_matmul_i8(lhs, rhs, output, metadata.as_ptr()),
                    "hodu_cpu_matmul_i16" => hodu_cpu_matmul_i16(lhs, rhs, output, metadata.as_ptr()),
                    "hodu_cpu_matmul_i32" => hodu_cpu_matmul_i32(lhs, rhs, output, metadata.as_ptr()),
                    "hodu_cpu_matmul_i64" => hodu_cpu_matmul_i64(lhs, rhs, output, metadata.as_ptr()),
                    _ => panic!("Unsupported matmul kernel: {}", name),
                }
            }

            // Dispatch function for dot
            unsafe fn dispatch_dot(
                name: &str,
                lhs: *const c_void,
                rhs: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    "hodu_cpu_dot_f8e4m3" => hodu_cpu_dot_f8e4m3(lhs, rhs, output, metadata),
                    "hodu_cpu_dot_f8e5m2" => hodu_cpu_dot_f8e5m2(lhs, rhs, output, metadata),
                    "hodu_cpu_dot_bf16" => hodu_cpu_dot_bf16(lhs, rhs, output, metadata),
                    "hodu_cpu_dot_f16" => hodu_cpu_dot_f16(lhs, rhs, output, metadata),
                    "hodu_cpu_dot_f32" => hodu_cpu_dot_f32(lhs, rhs, output, metadata),
                    "hodu_cpu_dot_f64" => hodu_cpu_dot_f64(lhs, rhs, output, metadata),
                    "hodu_cpu_dot_u8" => hodu_cpu_dot_u8(lhs, rhs, output, metadata),
                    "hodu_cpu_dot_u16" => hodu_cpu_dot_u16(lhs, rhs, output, metadata),
                    "hodu_cpu_dot_u32" => hodu_cpu_dot_u32(lhs, rhs, output, metadata),
                    "hodu_cpu_dot_u64" => hodu_cpu_dot_u64(lhs, rhs, output, metadata),
                    "hodu_cpu_dot_i8" => hodu_cpu_dot_i8(lhs, rhs, output, metadata),
                    "hodu_cpu_dot_i16" => hodu_cpu_dot_i16(lhs, rhs, output, metadata),
                    "hodu_cpu_dot_i32" => hodu_cpu_dot_i32(lhs, rhs, output, metadata),
                    "hodu_cpu_dot_i64" => hodu_cpu_dot_i64(lhs, rhs, output, metadata),
                    _ => panic!("Unsupported dot kernel: {}", name),
                }
            }
        }
    };
}

// Declare matrix operations
declare_and_dispatch_matrix!(matmul, dot);
