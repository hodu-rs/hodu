//! Memory operations
//!
//! This module provides memory layout operations for tensors:
//! - contiguous: Copy strided tensor data to contiguous memory
//!
//! All operations support strided tensor access and multiple data types.

use crate::{error::Result, kernels::macros::ops};
use core::ffi::c_void;

// Define contiguous operation using the macro
ops!(contiguous);

/// Call contiguous operation by kernel name
///
/// Copies tensor data from potentially strided layout to contiguous memory.
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
pub fn call_ops_contiguous(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_contiguous(kernel_name.0, input, output, metadata.as_ptr());
    }

    Ok(())
}

// Macro to automatically generate extern declarations and dispatch for contiguous operation
macro_rules! declare_and_dispatch_contiguous {
    ($($type_suffix:ident),* $(,)?) => {
        paste::paste! {
            // Extern C declarations
            extern "C" {
                $(
                    fn [<hodu_cpu_contiguous_ $type_suffix>](
                        input: *const c_void,
                        output: *mut c_void,
                        metadata: *const usize
                    );
                )*
            }

            // Dispatch function
            unsafe fn dispatch_contiguous(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!("hodu_cpu_contiguous_", stringify!($type_suffix)) => {
                            [<hodu_cpu_contiguous_ $type_suffix>](input, output, metadata)
                        }
                    )*
                    _ => panic!("Unsupported contiguous kernel: {}", name),
                }
            }
        }
    };
}

// Declare all contiguous operations
declare_and_dispatch_contiguous!(bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
