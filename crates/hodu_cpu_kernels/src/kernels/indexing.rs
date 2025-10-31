use crate::{error::Result, kernels::macros::ops};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::ffi::c_void;

ops!(
    index_select,
    index_put,
    gather,
    scatter,
    scatter_add,
    scatter_max,
    scatter_min
);

macro_rules! declare_and_dispatch_index_select {
    ($($op:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<$op _bool>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f8e4m3>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f8e5m2>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _bf16>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f16>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f32>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f64>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i8>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i16>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i32>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i64>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u8>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u16>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u32>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u64>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                )*
            }

            unsafe fn dispatch_index_select(
                name: &str,
                input: *const c_void,
                indices: *const i32,
                output: *mut c_void,
                num_els: usize,
                num_dims: usize,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!(stringify!($op), "_bool") => [<$op _bool>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f8e4m3") => [<$op _f8e4m3>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f8e5m2") => [<$op _f8e5m2>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_bf16") => [<$op _bf16>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f16") => [<$op _f16>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f32") => [<$op _f32>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f64") => [<$op _f64>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i8") => [<$op _i8>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i16") => [<$op _i16>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i32") => [<$op _i32>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i64") => [<$op _i64>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u8") => [<$op _u8>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u16") => [<$op _u16>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u32") => [<$op _u32>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u64") => [<$op _u64>](input, indices, output, num_els, num_dims, metadata),
                    )*
                    _ => panic!("Unknown index_select operation: {}", name),
                }
            }
        }
    };
}

macro_rules! declare_and_dispatch_index_put {
    ($($op:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<$op _bool>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f8e4m3>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f8e5m2>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _bf16>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f16>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f32>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f64>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i8>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i16>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i32>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i64>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u8>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u16>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u32>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u64>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                )*
            }

            unsafe fn dispatch_index_put(
                name: &str,
                input: *const c_void,
                indices: *const i32,
                values: *const c_void,
                output: *mut c_void,
                num_els: usize,
                num_dims: usize,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!(stringify!($op), "_bool") => [<$op _bool>](input, indices, values, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f8e4m3") => [<$op _f8e4m3>](input, indices, values, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f8e5m2") => [<$op _f8e5m2>](input, indices, values, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_bf16") => [<$op _bf16>](input, indices, values, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f16") => [<$op _f16>](input, indices, values, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f32") => [<$op _f32>](input, indices, values, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f64") => [<$op _f64>](input, indices, values, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i8") => [<$op _i8>](input, indices, values, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i16") => [<$op _i16>](input, indices, values, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i32") => [<$op _i32>](input, indices, values, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i64") => [<$op _i64>](input, indices, values, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u8") => [<$op _u8>](input, indices, values, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u16") => [<$op _u16>](input, indices, values, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u32") => [<$op _u32>](input, indices, values, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u64") => [<$op _u64>](input, indices, values, output, num_els, num_dims, metadata),
                    )*
                    _ => panic!("Unknown index_put operation: {}", name),
                }
            }
        }
    };
}

macro_rules! declare_and_dispatch_gather {
    ($($op:ident),* $(,)?) => {
        paste::paste! {
            extern "C" {
                $(
                    fn [<$op _bool>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f8e4m3>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f8e5m2>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _bf16>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f16>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f32>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _f64>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i8>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i16>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i32>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _i64>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u8>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u16>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u32>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                    fn [<$op _u64>](input: *const c_void, indices: *const i32, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                )*
            }

            unsafe fn dispatch_gather(
                name: &str,
                input: *const c_void,
                indices: *const i32,
                output: *mut c_void,
                num_els: usize,
                num_dims: usize,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!(stringify!($op), "_bool") => [<$op _bool>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f8e4m3") => [<$op _f8e4m3>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f8e5m2") => [<$op _f8e5m2>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_bf16") => [<$op _bf16>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f16") => [<$op _f16>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f32") => [<$op _f32>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_f64") => [<$op _f64>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i8") => [<$op _i8>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i16") => [<$op _i16>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i32") => [<$op _i32>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_i64") => [<$op _i64>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u8") => [<$op _u8>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u16") => [<$op _u16>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u32") => [<$op _u32>](input, indices, output, num_els, num_dims, metadata),
                        concat!(stringify!($op), "_u64") => [<$op _u64>](input, indices, output, num_els, num_dims, metadata),
                    )*
                    _ => panic!("Unknown gather operation: {}", name),
                }
            }
        }
    };
}

macro_rules! declare_and_dispatch_scatter {
    ($op_name:ident, $dispatch_name:ident) => {
        paste::paste! {
            extern "C" {
                fn [<$op_name _bool>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _f8e4m3>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _f8e5m2>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _bf16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _f16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _f32>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _f64>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _i8>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _i16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _i32>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _i64>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _u8>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _u16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _u32>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _u64>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
            }

            unsafe fn $dispatch_name(
                name: &str,
                input: *const c_void,
                indices: *const i32,
                src: *const c_void,
                output: *mut c_void,
                num_els: usize,
                num_dims: usize,
                metadata: *const usize,
            ) {
                match name {
                    concat!(stringify!($op_name), "_bool") => [<$op_name _bool>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_f8e4m3") => [<$op_name _f8e4m3>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_f8e5m2") => [<$op_name _f8e5m2>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_bf16") => [<$op_name _bf16>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_f16") => [<$op_name _f16>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_f32") => [<$op_name _f32>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_f64") => [<$op_name _f64>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_i8") => [<$op_name _i8>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_i16") => [<$op_name _i16>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_i32") => [<$op_name _i32>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_i64") => [<$op_name _i64>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_u8") => [<$op_name _u8>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_u16") => [<$op_name _u16>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_u32") => [<$op_name _u32>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_u64") => [<$op_name _u64>](input, indices, src, output, num_els, num_dims, metadata),
                    _ => panic!("Unknown {} operation: {}", stringify!($op_name), name),
                }
            }
        }
    };
}

macro_rules! declare_and_dispatch_scatter_add {
    ($op_name:ident, $dispatch_name:ident) => {
        paste::paste! {
            extern "C" {
                fn [<$op_name _f8e4m3>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _f8e5m2>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _bf16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _f16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _f32>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _i8>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _i16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _i32>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _i64>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _u8>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _u16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _u32>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
                fn [<$op_name _u64>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, num_els: usize, num_dims: usize, metadata: *const usize);
            }

            unsafe fn $dispatch_name(
                name: &str,
                input: *const c_void,
                indices: *const i32,
                src: *const c_void,
                output: *mut c_void,
                num_els: usize,
                num_dims: usize,
                metadata: *const usize,
            ) {
                match name {
                    concat!(stringify!($op_name), "_f8e4m3") => [<$op_name _f8e4m3>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_f8e5m2") => [<$op_name _f8e5m2>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_bf16") => [<$op_name _bf16>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_f16") => [<$op_name _f16>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_f32") => [<$op_name _f32>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_i8") => [<$op_name _i8>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_i16") => [<$op_name _i16>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_i32") => [<$op_name _i32>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_i64") => [<$op_name _i64>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_u8") => [<$op_name _u8>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_u16") => [<$op_name _u16>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_u32") => [<$op_name _u32>](input, indices, src, output, num_els, num_dims, metadata),
                    concat!(stringify!($op_name), "_u64") => [<$op_name _u64>](input, indices, src, output, num_els, num_dims, metadata),
                    _ => panic!("Unknown {} operation: {}", stringify!($op_name), name),
                }
            }
        }
    };
}

declare_and_dispatch_index_select!(index_select);
declare_and_dispatch_index_put!(index_put);
declare_and_dispatch_gather!(gather);
declare_and_dispatch_scatter!(scatter, dispatch_scatter);
declare_and_dispatch_scatter_add!(scatter_add, dispatch_scatter_add);
declare_and_dispatch_scatter_add!(scatter_max, dispatch_scatter_max);
declare_and_dispatch_scatter_add!(scatter_min, dispatch_scatter_min);

#[allow(clippy::too_many_arguments)]
pub fn call_index_select(
    kernel_name: crate::kernels::macros::Kernel,
    shape: &[usize],
    input: *const c_void,
    input_strides: &[usize],
    input_offset: usize,
    indices: *const i32,
    dim: usize,
    num_indices: usize,
    output: *mut c_void,
) -> Result<()> {
    let num_dims = shape.len();

    let mut output_shape = shape.to_vec();
    output_shape[dim] = num_indices;
    let num_els: usize = output_shape.iter().product();

    let mut metadata = Vec::with_capacity(num_dims * 2 + 3);
    metadata.extend_from_slice(shape);
    metadata.extend_from_slice(input_strides);
    metadata.push(input_offset);
    metadata.push(dim);
    metadata.push(num_indices);

    unsafe {
        dispatch_index_select(
            kernel_name.0,
            input,
            indices,
            output,
            num_els,
            num_dims,
            metadata.as_ptr(),
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_index_put(
    kernel_name: crate::kernels::macros::Kernel,
    input_shape: &[usize],
    input: *const c_void,
    input_strides: &[usize],
    input_offset: usize,
    indices: *const i32,
    values: *const c_void,
    values_strides: &[usize],
    values_offset: usize,
    dim: usize,
    num_indices: usize,
    output: *mut c_void,
) -> Result<()> {
    let num_dims = input_shape.len();
    let num_els: usize = input_shape.iter().product();

    let mut metadata = Vec::with_capacity(num_dims * 3 + 4);
    metadata.extend_from_slice(input_shape);
    metadata.extend_from_slice(input_strides);
    metadata.extend_from_slice(values_strides);
    metadata.push(input_offset);
    metadata.push(values_offset);
    metadata.push(dim);
    metadata.push(num_indices);

    unsafe {
        dispatch_index_put(
            kernel_name.0,
            input,
            indices,
            values,
            output,
            num_els,
            num_dims,
            metadata.as_ptr(),
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_gather(
    kernel_name: crate::kernels::macros::Kernel,
    input_shape: &[usize],
    input: *const c_void,
    input_strides: &[usize],
    input_offset: usize,
    indices: *const i32,
    indices_strides: &[usize],
    indices_offset: usize,
    indices_len: usize,
    dim: usize,
    output: *mut c_void,
) -> Result<()> {
    let num_dims = input_shape.len();
    let num_els: usize = input_shape
        .iter()
        .enumerate()
        .map(|(i, &s)| if i == dim { indices_len } else { s })
        .product();

    let mut metadata = Vec::with_capacity(num_dims * 3 + 4);
    metadata.extend_from_slice(input_shape);
    metadata.extend_from_slice(input_strides);
    metadata.extend_from_slice(indices_strides);
    metadata.push(input_offset);
    metadata.push(indices_offset);
    metadata.push(dim);
    metadata.push(indices_len);

    unsafe {
        dispatch_gather(
            kernel_name.0,
            input,
            indices,
            output,
            num_els,
            num_dims,
            metadata.as_ptr(),
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_scatter(
    kernel_name: crate::kernels::macros::Kernel,
    input_shape: &[usize],
    input: *const c_void,
    input_strides: &[usize],
    input_offset: usize,
    indices: *const i32,
    indices_strides: &[usize],
    indices_offset: usize,
    src: *const c_void,
    src_shape: &[usize],
    src_strides: &[usize],
    src_offset: usize,
    dim: usize,
    output: *mut c_void,
) -> Result<()> {
    let num_dims = input_shape.len();
    let num_els: usize = src_shape.iter().product();

    let mut metadata = Vec::with_capacity(num_dims * 5 + 4);
    metadata.extend_from_slice(input_shape);
    metadata.extend_from_slice(input_strides);
    metadata.extend_from_slice(src_shape);
    metadata.extend_from_slice(src_strides);
    metadata.extend_from_slice(indices_strides);
    metadata.push(input_offset);
    metadata.push(src_offset);
    metadata.push(indices_offset);
    metadata.push(dim);

    unsafe {
        let name = kernel_name.0;
        if name.starts_with("scatter_add_") {
            dispatch_scatter_add(name, input, indices, src, output, num_els, num_dims, metadata.as_ptr());
        } else if name.starts_with("scatter_max_") {
            dispatch_scatter_max(name, input, indices, src, output, num_els, num_dims, metadata.as_ptr());
        } else if name.starts_with("scatter_min_") {
            dispatch_scatter_min(name, input, indices, src, output, num_els, num_dims, metadata.as_ptr());
        } else {
            dispatch_scatter(name, input, indices, src, output, num_els, num_dims, metadata.as_ptr());
        }
    }

    Ok(())
}
