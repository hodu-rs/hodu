//! Binary operations for tensor element-wise computations
//!
//! This module provides element-wise binary operations including:
//! - Arithmetic: add, sub, mul, div, pow, maximum, minimum
//! - Logical: logical_and, logical_or, logical_xor
//! - Comparison: eq, ne, lt, le, gt, ge
//!
//! All operations support multiple data types including floating point
//! (f8e4m3, f8e5m2, bf16, f16, f32, f64), integer (u8-u64, i8-i64), and bool.

use crate::{error::Result, kernels::macros::ops};
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

/// Execute a binary operation on two tensors
///
/// Performs element-wise binary operations on tensors with arbitrary shapes and strides.
/// The function automatically handles contiguous and strided memory layouts for optimal
/// performance. Broadcasting is not handled here - shapes must be compatible.
///
/// # Arguments
/// * `kernel_name` - The binary operation to perform (e.g., add::F32, mul::I32)
/// * `lhs` - Pointer to left-hand side tensor data
/// * `rhs` - Pointer to right-hand side tensor data
/// * `output` - Pointer to output tensor buffer
/// * `metadata` - Tensor metadata array (see layout below)
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements to process)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: lhs_shape (shape of left tensor)
/// - metadata[2+num_dims..2+2*num_dims]: rhs_shape (shape of right tensor)
/// - metadata[2+2*num_dims..2+3*num_dims]: lhs_strides (stride of left tensor)
/// - metadata[2+3*num_dims..2+4*num_dims]: rhs_strides (stride of right tensor)
/// - metadata[2+4*num_dims]: lhs_offset (starting offset in left tensor)
/// - metadata[2+4*num_dims+1]: rhs_offset (starting offset in right tensor)
///
/// # Safety
/// This function uses unsafe FFI calls to C kernels. Caller must ensure:
/// - Pointers are valid and properly aligned
/// - Metadata accurately describes tensor layout
/// - Output buffer has sufficient capacity
///
/// # Returns
/// Returns `Ok(())` on success. Currently does not propagate C kernel errors.
pub fn call_ops_binary(
    kernel_name: crate::kernels::macros::Kernel,
    lhs: *const c_void,
    rhs: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_binary(kernel_name.0, lhs, rhs, output, metadata.as_ptr());
    }

    Ok(())
}

/// Macro to generate extern C declarations and dispatch logic for binary operations
///
/// This macro generates:
/// 1. Extern "C" function declarations for all operation/type combinations
/// 2. A dispatch function that routes kernel name strings to the appropriate C function
///
/// For each operation (e.g., add, mul), it creates declarations for all supported types
/// (bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64).
///
/// # Safety
/// The generated code involves unsafe FFI calls. Type safety is ensured by the
/// kernel naming convention (operation_type format, e.g., "add_f32").
macro_rules! declare_and_dispatch_binary {
    ($($op:ident),* $(,)?) => {
        paste::paste! {
            // Extern C declarations for all operations and types
            extern "C" {
                $(
                    fn [<hodu_cpu_ $op _bool>](lhs: *const c_void, rhs: *const c_void, output: *mut c_void, metadata: *const usize);
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

            // Dispatch function
            unsafe fn dispatch_binary(
                name: &str,
                lhs: *const c_void,
                rhs: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!("hodu_cpu_", stringify!($op), "_bool") => [<hodu_cpu_ $op _bool>](lhs, rhs, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_f8e4m3") => [<hodu_cpu_ $op _f8e4m3>](lhs, rhs, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_f8e5m2") => [<hodu_cpu_ $op _f8e5m2>](lhs, rhs, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_bf16") => [<hodu_cpu_ $op _bf16>](lhs, rhs, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_f16") => [<hodu_cpu_ $op _f16>](lhs, rhs, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_f32") => [<hodu_cpu_ $op _f32>](lhs, rhs, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_f64") => [<hodu_cpu_ $op _f64>](lhs, rhs, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_u8") => [<hodu_cpu_ $op _u8>](lhs, rhs, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_u16") => [<hodu_cpu_ $op _u16>](lhs, rhs, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_u32") => [<hodu_cpu_ $op _u32>](lhs, rhs, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_u64") => [<hodu_cpu_ $op _u64>](lhs, rhs, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_i8") => [<hodu_cpu_ $op _i8>](lhs, rhs, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_i16") => [<hodu_cpu_ $op _i16>](lhs, rhs, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_i32") => [<hodu_cpu_ $op _i32>](lhs, rhs, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_i64") => [<hodu_cpu_ $op _i64>](lhs, rhs, output, metadata),
                    )*
                    _ => panic!("Unsupported binary kernel: {}", name),
                }
            }

            /// Get function pointer for a binary kernel by name
            /// Returns None if kernel name is not recognized
            pub fn get_binary_kernel_ptr(name: &str) -> Option<*const ()> {
                match name {
                    $(
                        concat!("hodu_cpu_", stringify!($op), "_bool") => Some([<hodu_cpu_ $op _bool>] as *const ()),
                        concat!("hodu_cpu_", stringify!($op), "_f8e4m3") => Some([<hodu_cpu_ $op _f8e4m3>] as *const ()),
                        concat!("hodu_cpu_", stringify!($op), "_f8e5m2") => Some([<hodu_cpu_ $op _f8e5m2>] as *const ()),
                        concat!("hodu_cpu_", stringify!($op), "_bf16") => Some([<hodu_cpu_ $op _bf16>] as *const ()),
                        concat!("hodu_cpu_", stringify!($op), "_f16") => Some([<hodu_cpu_ $op _f16>] as *const ()),
                        concat!("hodu_cpu_", stringify!($op), "_f32") => Some([<hodu_cpu_ $op _f32>] as *const ()),
                        concat!("hodu_cpu_", stringify!($op), "_f64") => Some([<hodu_cpu_ $op _f64>] as *const ()),
                        concat!("hodu_cpu_", stringify!($op), "_u8") => Some([<hodu_cpu_ $op _u8>] as *const ()),
                        concat!("hodu_cpu_", stringify!($op), "_u16") => Some([<hodu_cpu_ $op _u16>] as *const ()),
                        concat!("hodu_cpu_", stringify!($op), "_u32") => Some([<hodu_cpu_ $op _u32>] as *const ()),
                        concat!("hodu_cpu_", stringify!($op), "_u64") => Some([<hodu_cpu_ $op _u64>] as *const ()),
                        concat!("hodu_cpu_", stringify!($op), "_i8") => Some([<hodu_cpu_ $op _i8>] as *const ()),
                        concat!("hodu_cpu_", stringify!($op), "_i16") => Some([<hodu_cpu_ $op _i16>] as *const ()),
                        concat!("hodu_cpu_", stringify!($op), "_i32") => Some([<hodu_cpu_ $op _i32>] as *const ()),
                        concat!("hodu_cpu_", stringify!($op), "_i64") => Some([<hodu_cpu_ $op _i64>] as *const ()),
                    )*
                    _ => None,
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
