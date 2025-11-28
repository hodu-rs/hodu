//! Storage operations
//!
//! This module provides storage-level operations for tensors:
//! - const_set: Fill tensor with a constant value
//!
//! All operations support strided tensor access and multiple data types.

use crate::{error::Result, kernels::macros::ops};
use core::ffi::c_void;

// Define const_set operation using the macro
ops!(const_set);

/// Call const_set operation by kernel name
///
/// Fills a tensor with a constant value.
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape
/// - metadata[2+num_dims..2+2*num_dims]: strides
/// - metadata[2+2*num_dims]: offset
///
/// # Safety
/// - `output` must point to a valid output buffer with sufficient capacity
/// - `value` must be of the same type as the tensor elements
/// - Metadata must accurately describe the tensor layout
pub fn call_ops_const_set<T>(
    kernel_name: crate::kernels::macros::Kernel,
    output: *mut c_void,
    metadata: &[usize],
    value: T,
) -> Result<()> {
    unsafe {
        dispatch_const_set(
            kernel_name.0,
            output,
            metadata.as_ptr(),
            &value as *const T as *const c_void,
        );
    }

    Ok(())
}

// Macro to automatically generate extern declarations and dispatch for const_set operation
macro_rules! declare_and_dispatch_const_set {
    ($($type_suffix:ident),* $(,)?) => {
        paste::paste! {
            // Extern C declarations
            extern "C" {
                $(
                    fn [<hodu_cpu_const_set_ $type_suffix>](
                        output: *mut c_void,
                        metadata: *const usize,
                        value: *const c_void
                    );
                )*
            }

            // Dispatch function
            unsafe fn dispatch_const_set(
                name: &str,
                output: *mut c_void,
                metadata: *const usize,
                value: *const c_void,
            ) {
                match name {
                    $(
                        concat!("hodu_cpu_const_set_", stringify!($type_suffix)) => {
                            [<hodu_cpu_const_set_ $type_suffix>](output, metadata, value)
                        }
                    )*
                    _ => panic!("Unsupported const_set kernel: {}", name),
                }
            }
        }
    };
}

// Declare all const_set operations
declare_and_dispatch_const_set!(bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
