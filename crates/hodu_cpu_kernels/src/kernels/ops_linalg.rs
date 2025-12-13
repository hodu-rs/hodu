//! Linear algebra operations
//!
//! This module provides:
//! - `det`: Matrix determinant computation using LU decomposition

use crate::{error::Result, kernels::macros::ops};
use core::ffi::c_void;

// Define all linalg operations using the macro
ops!(det);

/// Execute a matrix determinant operation
///
/// Computes the determinant of square matrices with optional batch dimensions.
/// Uses LU decomposition for general NxN matrices, direct formulas for 2x2 and 3x3.
///
/// # Arguments
/// * `kernel_name` - The det kernel to execute (e.g., det::F32)
/// * `input` - Pointer to input tensor
/// * `output` - Pointer to output tensor buffer
/// * `metadata` - Tensor metadata array (see layout below)
///
/// # Metadata layout
/// - metadata[0]: batch_size (product of batch dimensions)
/// - metadata[1]: n (matrix size, N×N)
/// - metadata[2]: ndim (total number of dimensions)
/// - metadata[3..3+ndim]: shape
/// - metadata[3+ndim..3+2*ndim]: strides
/// - metadata[3+2*ndim]: offset
///
/// # Safety
/// This function uses unsafe FFI calls to C kernels. Caller must ensure:
/// - All pointers are valid and properly aligned
/// - Metadata accurately describes tensor layout
/// - Output buffer has sufficient capacity (batch_size elements)
/// - Input is a square matrix: [..., N, N] -> [...]
///
/// # Returns
/// Returns `Ok(())` on success.
pub fn call_ops_det(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_det(kernel_name.0, input, output, metadata.as_ptr());
    }

    Ok(())
}

// Det extern C declarations
extern "C" {
    fn hodu_cpu_det_f8e4m3(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_det_f8e5m2(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_det_bf16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_det_f16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_det_f32(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_det_f64(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_det_u8(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_det_u16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_det_u32(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_det_u64(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_det_i8(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_det_i16(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_det_i32(input: *const c_void, output: *mut c_void, metadata: *const usize);
    fn hodu_cpu_det_i64(input: *const c_void, output: *mut c_void, metadata: *const usize);
}

unsafe fn dispatch_det(name: &str, input: *const c_void, output: *mut c_void, metadata: *const usize) {
    match name {
        "hodu_cpu_det_f8e4m3" => hodu_cpu_det_f8e4m3(input, output, metadata),
        "hodu_cpu_det_f8e5m2" => hodu_cpu_det_f8e5m2(input, output, metadata),
        "hodu_cpu_det_bf16" => hodu_cpu_det_bf16(input, output, metadata),
        "hodu_cpu_det_f16" => hodu_cpu_det_f16(input, output, metadata),
        "hodu_cpu_det_f32" => hodu_cpu_det_f32(input, output, metadata),
        "hodu_cpu_det_f64" => hodu_cpu_det_f64(input, output, metadata),
        "hodu_cpu_det_u8" => hodu_cpu_det_u8(input, output, metadata),
        "hodu_cpu_det_u16" => hodu_cpu_det_u16(input, output, metadata),
        "hodu_cpu_det_u32" => hodu_cpu_det_u32(input, output, metadata),
        "hodu_cpu_det_u64" => hodu_cpu_det_u64(input, output, metadata),
        "hodu_cpu_det_i8" => hodu_cpu_det_i8(input, output, metadata),
        "hodu_cpu_det_i16" => hodu_cpu_det_i16(input, output, metadata),
        "hodu_cpu_det_i32" => hodu_cpu_det_i32(input, output, metadata),
        "hodu_cpu_det_i64" => hodu_cpu_det_i64(input, output, metadata),
        _ => panic!("Unsupported det kernel: {}", name),
    }
}
