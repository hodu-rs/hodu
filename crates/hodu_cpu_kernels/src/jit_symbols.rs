//! JIT symbol resolution
//!
//! This module provides function pointers to C kernel functions
//! for JIT execution engines

/// Get function pointer for any kernel by name
/// Returns None if kernel name is not recognized
///
/// This function can resolve all kernel names in the format:
/// "hodu_cpu_{operation}_{dtype}"
///
/// Examples:
/// - "hodu_cpu_add_f32"
/// - "hodu_cpu_mul_i32"
/// - "hodu_cpu_matmul_f64"
pub fn get_kernel_ptr(name: &str) -> Option<*const ()> {
    // Try each operation category
    if let Some(ptr) = crate::kernels::ops_binary::get_binary_kernel_ptr(name) {
        return Some(ptr);
    }

    // TODO: Add other operation categories as we implement them:
    // - ops_unary::get_unary_kernel_ptr(name)
    // - ops_matrix::get_matrix_kernel_ptr(name)
    // - ops_reduce::get_reduce_kernel_ptr(name)
    // - etc.

    None
}
