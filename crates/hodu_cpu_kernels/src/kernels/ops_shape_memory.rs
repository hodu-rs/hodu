//! Shape memory operations
//!
//! This module provides shape-related operations that require memory copy:
//! - flip: Reverse tensor along specified dimensions
//!
//! All operations support multiple data types.

use crate::{error::Result, kernels::macros::ops};
use core::ffi::c_void;

ops!(flip);

/// Call flip operation by kernel name
///
/// Reverses tensor along specified dimensions.
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape
/// - metadata[2+num_dims..2+2*num_dims]: flip_mask (1 = flip this dim, 0 = don't flip)
///
/// # Safety
/// - `input` must point to valid tensor data of the appropriate type
/// - `output` must point to a valid output buffer with sufficient capacity
/// - Metadata must accurately describe the tensor layout
pub fn call_ops_flip(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_flip(kernel_name.0, input, output, metadata.as_ptr());
    }
    Ok(())
}

macro_rules! declare_and_dispatch_flip {
    ($($dtype:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<hodu_cpu_flip_ $dtype>](
                        input: *const c_void,
                        output: *mut c_void,
                        metadata: *const usize,
                    );
                )*
            }

            unsafe fn dispatch_flip(
                kernel_name: &str,
                input: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match kernel_name {
                    $(
                        concat!("hodu_cpu_flip_", stringify!($dtype)) => {
                            [<hodu_cpu_flip_ $dtype>](input, output, metadata)
                        }
                    )*
                    _ => panic!("Unknown kernel: {}", kernel_name),
                }
            }
        }
    };
}

declare_and_dispatch_flip!(bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
