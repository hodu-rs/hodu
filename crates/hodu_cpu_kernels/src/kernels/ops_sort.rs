//! Sort operations
//!
//! This module provides sorting operations:
//! - topk: Get top-k largest or smallest elements along a dimension
//!
//! All operations support multiple data types.

use crate::{error::Result, kernels::macros::ops};
use core::ffi::c_void;

ops!(topk);

/// Call topk operation by kernel name
///
/// Returns the k largest or smallest elements along the last dimension.
///
/// # Metadata layout
/// - metadata[0]: output_size (k * outer_size)
/// - metadata[1]: k (number of top elements)
/// - metadata[2]: last_dim_size (size of the dimension to search along)
/// - metadata[3]: outer_size (product of all dimensions except last)
/// - metadata[4]: largest (1 = largest, 0 = smallest)
/// - metadata[5]: sorted (1 = sorted, 0 = unsorted)
/// - metadata[6]: offset
///
/// # Safety
/// - `input` must point to valid tensor data of the appropriate type
/// - `values` must point to a valid output buffer for values
/// - `indices` must point to a valid output buffer for indices (i32)
/// - Metadata must accurately describe the tensor layout
pub fn call_topk(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    values: *mut c_void,
    indices: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_topk(kernel_name.0, input, values, indices, metadata.as_ptr());
    }
    Ok(())
}

macro_rules! declare_and_dispatch_topk {
    ($($dtype:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<hodu_cpu_topk_ $dtype>](
                        input: *const c_void,
                        values: *mut c_void,
                        indices: *mut c_void,
                        metadata: *const usize,
                    );
                )*
            }

            unsafe fn dispatch_topk(
                kernel_name: &str,
                input: *const c_void,
                values: *mut c_void,
                indices: *mut c_void,
                metadata: *const usize,
            ) {
                match kernel_name {
                    $(
                        concat!("hodu_cpu_topk_", stringify!($dtype)) => {
                            [<hodu_cpu_topk_ $dtype>](input, values, indices, metadata)
                        }
                    )*
                    _ => panic!("Unknown kernel: {}", kernel_name),
                }
            }
        }
    };
}

declare_and_dispatch_topk!(f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
