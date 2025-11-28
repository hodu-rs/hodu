//! Cast operations
//!
//! This module provides type casting operations for tensors.
//! Supports conversion between all data types (15x15 combinations).
//!
//! All operations support strided tensor access.

use crate::error::Result;
use core::ffi::c_void;

/// Kernel identifier for cast operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CastKernel(pub &'static str);

/// Cast kernel constants for each source-destination type pair
pub mod cast {
    use super::CastKernel;

    macro_rules! define_cast_kernels {
        ($from:ident => $($to:ident),+ $(,)?) => {
            paste::paste! {
                pub mod [<from_ $from>] {
                    use super::CastKernel;
                    $(
                        pub const [<TO_ $to:upper>]: CastKernel = CastKernel(
                            concat!("hodu_cpu_cast_", stringify!($from), "_to_", stringify!($to))
                        );
                    )+
                }
            }
        };
    }

    define_cast_kernels!(bool => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
    define_cast_kernels!(f8e4m3 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
    define_cast_kernels!(f8e5m2 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
    define_cast_kernels!(bf16 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
    define_cast_kernels!(f16 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
    define_cast_kernels!(f32 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
    define_cast_kernels!(f64 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
    define_cast_kernels!(u8 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
    define_cast_kernels!(u16 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
    define_cast_kernels!(u32 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
    define_cast_kernels!(u64 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
    define_cast_kernels!(i8 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
    define_cast_kernels!(i16 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
    define_cast_kernels!(i32 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
    define_cast_kernels!(i64 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64);
}

/// Call cast operation by kernel name
///
/// Converts tensor elements from one data type to another.
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape
/// - metadata[2+num_dims..2+2*num_dims]: strides
/// - metadata[2+2*num_dims]: offset
///
/// # Safety
/// - `input` must point to valid tensor data of the source type
/// - `output` must point to a valid output buffer with sufficient capacity for the destination type
/// - Metadata must accurately describe the tensor layout
pub fn call_ops_cast(
    kernel_name: CastKernel,
    input: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_cast(kernel_name.0, input, output, metadata.as_ptr());
    }

    Ok(())
}

// Macro to generate extern declarations and dispatch for all cast combinations
macro_rules! declare_and_dispatch_cast {
    ($($from:ident => $($to:ident),+);* $(;)?) => {
        paste::paste! {
            // Extern C declarations
            extern "C" {
                $($(
                    fn [<hodu_cpu_cast_ $from _to_ $to>](
                        input: *const c_void,
                        output: *mut c_void,
                        metadata: *const usize
                    );
                )+)*
            }

            // Dispatch function
            unsafe fn dispatch_cast(
                name: &str,
                input: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    $($(
                        concat!("hodu_cpu_cast_", stringify!($from), "_to_", stringify!($to)) => {
                            [<hodu_cpu_cast_ $from _to_ $to>](input, output, metadata)
                        }
                    )+)*
                    _ => panic!("Unsupported cast kernel: {}", name),
                }
            }
        }
    };
}

// Declare all cast operations (15x15 = 225 combinations)
declare_and_dispatch_cast!(
    bool => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64;
    f8e4m3 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64;
    f8e5m2 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64;
    bf16 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64;
    f16 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64;
    f32 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64;
    f64 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64;
    u8 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64;
    u16 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64;
    u32 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64;
    u64 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64;
    i8 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64;
    i16 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64;
    i32 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64;
    i64 => bool, f8e4m3, f8e5m2, bf16, f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64;
);
