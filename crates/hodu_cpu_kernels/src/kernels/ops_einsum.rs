//! Einsum operations
//!
//! This module provides Einstein summation operations for tensor contractions.

use crate::{error::Result, kernels::macros::ops};
use core::ffi::c_void;

ops!(einsum);

/// Call einsum operation by kernel name
///
/// # Metadata layout
/// Header:
/// - metadata[0]: num_output_els
/// - metadata[1]: num_inputs
/// - metadata[2]: num_total_indices
/// - metadata[3]: num_contraction_indices
/// - metadata[4]: output_ndim
/// - metadata[5..5+output_ndim]: output_shape
///
/// Per-input section (repeated num_inputs times):
/// - input_ndim
/// - input_shape[input_ndim]
/// - input_strides[input_ndim]
/// - input_offset
/// - index_to_dim_map[num_total_indices]
///
/// Index info:
/// - contraction_index_ids[num_contraction_indices]
/// - index_sizes[num_total_indices]
/// - output_index_ids[output_ndim]
pub fn call_ops_einsum(
    kernel_name: crate::kernels::macros::Kernel,
    inputs: &[*const c_void],
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_einsum(kernel_name.0, inputs.as_ptr(), output, metadata.as_ptr());
    }
    Ok(())
}

macro_rules! declare_and_dispatch_einsum {
    ($($dtype:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<hodu_cpu_einsum_ $dtype>](
                        inputs: *const *const c_void,
                        output: *mut c_void,
                        metadata: *const usize,
                    );
                )*
            }

            unsafe fn dispatch_einsum(
                kernel_name: &str,
                inputs: *const *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match kernel_name {
                    $(
                        concat!("hodu_cpu_einsum_", stringify!($dtype)) => {
                            [<hodu_cpu_einsum_ $dtype>](inputs, output, metadata)
                        }
                    )*
                    _ => panic!("Unknown kernel: {}", kernel_name),
                }
            }
        }
    };
}

declare_and_dispatch_einsum!(f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
