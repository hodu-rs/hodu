//! Indexing operations for tensor element selection and manipulation
//!
//! This module provides:
//! - `index_select`: Select elements along a dimension using indices
//! - `index_put`: Write values to specific indices
//! - `gather`: Gather elements using an indices tensor
//! - `scatter`: Scatter values to specific positions
//! - `scatter_add`: Scatter and accumulate values
//! - `scatter_max`: Scatter taking maximum values
//! - `scatter_min`: Scatter taking minimum values
//!
//! All operations support negative indexing (Python-style).

use crate::{error::Result, kernels::macros::ops};
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
                    fn [<hodu_cpu_ $op _bool>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f8e4m3>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f8e5m2>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _bf16>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f16>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f32>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f64>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i8>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i16>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i32>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i64>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u8>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u16>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u32>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u64>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                )*
            }

            unsafe fn dispatch_index_select(
                name: &str,
                input: *const c_void,
                indices: *const i32,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!("hodu_cpu_", stringify!($op), "_bool") => [<hodu_cpu_ $op _bool>](input, indices, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_f8e4m3") => [<hodu_cpu_ $op _f8e4m3>](input, indices, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_f8e5m2") => [<hodu_cpu_ $op _f8e5m2>](input, indices, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_bf16") => [<hodu_cpu_ $op _bf16>](input, indices, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_f16") => [<hodu_cpu_ $op _f16>](input, indices, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_f32") => [<hodu_cpu_ $op _f32>](input, indices, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_f64") => [<hodu_cpu_ $op _f64>](input, indices, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i8") => [<hodu_cpu_ $op _i8>](input, indices, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i16") => [<hodu_cpu_ $op _i16>](input, indices, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i32") => [<hodu_cpu_ $op _i32>](input, indices, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i64") => [<hodu_cpu_ $op _i64>](input, indices, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_u8") => [<hodu_cpu_ $op _u8>](input, indices, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_u16") => [<hodu_cpu_ $op _u16>](input, indices, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_u32") => [<hodu_cpu_ $op _u32>](input, indices, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_u64") => [<hodu_cpu_ $op _u64>](input, indices, output, metadata),
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
                    fn [<hodu_cpu_ $op _bool>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f8e4m3>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f8e5m2>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _bf16>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f16>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f32>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f64>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i8>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i16>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i32>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i64>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u8>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u16>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u32>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u64>](input: *const c_void, indices: *const i32, values: *const c_void, output: *mut c_void, metadata: *const usize);
                )*
            }

            unsafe fn dispatch_index_put(
                name: &str,
                input: *const c_void,
                indices: *const i32,
                values: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!("hodu_cpu_", stringify!($op), "_bool") => [<hodu_cpu_ $op _bool>](input, indices, values, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_f8e4m3") => [<hodu_cpu_ $op _f8e4m3>](input, indices, values, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_f8e5m2") => [<hodu_cpu_ $op _f8e5m2>](input, indices, values, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_bf16") => [<hodu_cpu_ $op _bf16>](input, indices, values, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_f16") => [<hodu_cpu_ $op _f16>](input, indices, values, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_f32") => [<hodu_cpu_ $op _f32>](input, indices, values, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_f64") => [<hodu_cpu_ $op _f64>](input, indices, values, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i8") => [<hodu_cpu_ $op _i8>](input, indices, values, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i16") => [<hodu_cpu_ $op _i16>](input, indices, values, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i32") => [<hodu_cpu_ $op _i32>](input, indices, values, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_i64") => [<hodu_cpu_ $op _i64>](input, indices, values, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_u8") => [<hodu_cpu_ $op _u8>](input, indices, values, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_u16") => [<hodu_cpu_ $op _u16>](input, indices, values, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_u32") => [<hodu_cpu_ $op _u32>](input, indices, values, output, metadata),
                        concat!("hodu_cpu_", stringify!($op), "_u64") => [<hodu_cpu_ $op _u64>](input, indices, values, output, metadata),
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
                    fn [<hodu_cpu_ $op _bool>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f8e4m3>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f8e5m2>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _bf16>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f16>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f32>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _f64>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i8>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i16>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i32>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _i64>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u8>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u16>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u32>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                    fn [<hodu_cpu_ $op _u64>](input: *const c_void, indices: *const i32, output: *mut c_void, metadata: *const usize);
                )*
            }

            unsafe fn dispatch_gather(
                name: &str,
                input: *const c_void,
                indices: *const i32,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    $(
                        concat!("hodu_cpu_", stringify!($op), "_bool") => [<hodu_cpu_ $op _bool>](input, indices, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_f8e4m3") => [<hodu_cpu_ $op _f8e4m3>](input, indices, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_f8e5m2") => [<hodu_cpu_ $op _f8e5m2>](input, indices, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_bf16") => [<hodu_cpu_ $op _bf16>](input, indices, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_f16") => [<hodu_cpu_ $op _f16>](input, indices, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_f32") => [<hodu_cpu_ $op _f32>](input, indices, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_f64") => [<hodu_cpu_ $op _f64>](input, indices, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_i8") => [<hodu_cpu_ $op _i8>](input, indices, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_i16") => [<hodu_cpu_ $op _i16>](input, indices, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_i32") => [<hodu_cpu_ $op _i32>](input, indices, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_i64") => [<hodu_cpu_ $op _i64>](input, indices, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_u8") => [<hodu_cpu_ $op _u8>](input, indices, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_u16") => [<hodu_cpu_ $op _u16>](input, indices, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_u32") => [<hodu_cpu_ $op _u32>](input, indices, output, metadata),
                            concat!("hodu_cpu_", stringify!($op), "_u64") => [<hodu_cpu_ $op _u64>](input, indices, output, metadata),
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
                fn [<hodu_cpu_ $op_name _bool>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _f8e4m3>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _f8e5m2>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _bf16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _f16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _f32>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _f64>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _i8>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _i16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _i32>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _i64>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _u8>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _u16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _u32>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _u64>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
            }

            unsafe fn $dispatch_name(
                name: &str,
                input: *const c_void,
                indices: *const i32,
                src: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    concat!("hodu_cpu_", stringify!($op_name), "_bool") => [<hodu_cpu_ $op_name _bool>](input, indices, src, output, metadata),
                        concat!("hodu_cpu_", stringify!($op_name), "_f8e4m3") => [<hodu_cpu_ $op_name _f8e4m3>](input, indices, src, output, metadata),
                        concat!("hodu_cpu_", stringify!($op_name), "_f8e5m2") => [<hodu_cpu_ $op_name _f8e5m2>](input, indices, src, output, metadata),
                        concat!("hodu_cpu_", stringify!($op_name), "_bf16") => [<hodu_cpu_ $op_name _bf16>](input, indices, src, output, metadata),
                        concat!("hodu_cpu_", stringify!($op_name), "_f16") => [<hodu_cpu_ $op_name _f16>](input, indices, src, output, metadata),
                        concat!("hodu_cpu_", stringify!($op_name), "_f32") => [<hodu_cpu_ $op_name _f32>](input, indices, src, output, metadata),
                        concat!("hodu_cpu_", stringify!($op_name), "_f64") => [<hodu_cpu_ $op_name _f64>](input, indices, src, output, metadata),
                        concat!("hodu_cpu_", stringify!($op_name), "_i8") => [<hodu_cpu_ $op_name _i8>](input, indices, src, output, metadata),
                        concat!("hodu_cpu_", stringify!($op_name), "_i16") => [<hodu_cpu_ $op_name _i16>](input, indices, src, output, metadata),
                        concat!("hodu_cpu_", stringify!($op_name), "_i32") => [<hodu_cpu_ $op_name _i32>](input, indices, src, output, metadata),
                        concat!("hodu_cpu_", stringify!($op_name), "_i64") => [<hodu_cpu_ $op_name _i64>](input, indices, src, output, metadata),
                        concat!("hodu_cpu_", stringify!($op_name), "_u8") => [<hodu_cpu_ $op_name _u8>](input, indices, src, output, metadata),
                        concat!("hodu_cpu_", stringify!($op_name), "_u16") => [<hodu_cpu_ $op_name _u16>](input, indices, src, output, metadata),
                        concat!("hodu_cpu_", stringify!($op_name), "_u32") => [<hodu_cpu_ $op_name _u32>](input, indices, src, output, metadata),
                        concat!("hodu_cpu_", stringify!($op_name), "_u64") => [<hodu_cpu_ $op_name _u64>](input, indices, src, output, metadata),
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
                fn [<hodu_cpu_ $op_name _f8e4m3>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _f8e5m2>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _bf16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _f16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _f32>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _i8>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _i16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _i32>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _i64>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _u8>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _u16>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _u32>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
                fn [<hodu_cpu_ $op_name _u64>](input: *const c_void, indices: *const i32, src: *const c_void, output: *mut c_void, metadata: *const usize);
            }

            unsafe fn $dispatch_name(
                name: &str,
                input: *const c_void,
                indices: *const i32,
                src: *const c_void,
                output: *mut c_void,
                metadata: *const usize,
            ) {
                match name {
                    concat!("hodu_cpu_", stringify!($op_name), "_f8e4m3") => [<hodu_cpu_ $op_name _f8e4m3>](input, indices, src, output, metadata),
                    concat!("hodu_cpu_", stringify!($op_name), "_f8e5m2") => [<hodu_cpu_ $op_name _f8e5m2>](input, indices, src, output, metadata),
                    concat!("hodu_cpu_", stringify!($op_name), "_bf16") => [<hodu_cpu_ $op_name _bf16>](input, indices, src, output, metadata),
                    concat!("hodu_cpu_", stringify!($op_name), "_f16") => [<hodu_cpu_ $op_name _f16>](input, indices, src, output, metadata),
                    concat!("hodu_cpu_", stringify!($op_name), "_f32") => [<hodu_cpu_ $op_name _f32>](input, indices, src, output, metadata),
                    concat!("hodu_cpu_", stringify!($op_name), "_i8") => [<hodu_cpu_ $op_name _i8>](input, indices, src, output, metadata),
                    concat!("hodu_cpu_", stringify!($op_name), "_i16") => [<hodu_cpu_ $op_name _i16>](input, indices, src, output, metadata),
                    concat!("hodu_cpu_", stringify!($op_name), "_i32") => [<hodu_cpu_ $op_name _i32>](input, indices, src, output, metadata),
                    concat!("hodu_cpu_", stringify!($op_name), "_i64") => [<hodu_cpu_ $op_name _i64>](input, indices, src, output, metadata),
                    concat!("hodu_cpu_", stringify!($op_name), "_u8") => [<hodu_cpu_ $op_name _u8>](input, indices, src, output, metadata),
                    concat!("hodu_cpu_", stringify!($op_name), "_u16") => [<hodu_cpu_ $op_name _u16>](input, indices, src, output, metadata),
                    concat!("hodu_cpu_", stringify!($op_name), "_u32") => [<hodu_cpu_ $op_name _u32>](input, indices, src, output, metadata),
                    concat!("hodu_cpu_", stringify!($op_name), "_u64") => [<hodu_cpu_ $op_name _u64>](input, indices, src, output, metadata),
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

/// Execute an index_select operation
///
/// Selects elements from the input tensor along a specified dimension using a 1D indices array.
/// Negative indices are supported (e.g., -1 refers to the last element).
///
/// # Arguments
/// * `kernel_name` - The index_select kernel to execute (e.g., index_select::F32)
/// * `input` - Pointer to input tensor data
/// * `indices` - Pointer to int32 indices array
/// * `output` - Pointer to output tensor buffer
/// * `metadata` - Tensor metadata array (see layout below)
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: shape (shape of input tensor)
/// - metadata[2+num_dims..2+2*num_dims]: input_strides (strides of input tensor)
/// - metadata[2+2*num_dims]: input_offset (starting offset in input tensor)
/// - metadata[2+2*num_dims+1]: dim (dimension along which to select)
/// - metadata[2+2*num_dims+2]: num_indices (number of indices to select)
///
/// # Safety
/// This function uses unsafe FFI calls to C kernels. Caller must ensure:
/// - All pointers are valid and properly aligned
/// - Metadata accurately describes tensor layout
/// - Output buffer has sufficient capacity (num_indices elements along dim)
///
/// # Returns
/// Returns `Ok(())` on success.
pub fn call_ops_index_select(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    indices: *const i32,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_index_select(kernel_name.0, input, indices, output, metadata.as_ptr());
    }

    Ok(())
}

/// Execute an index_put operation
///
/// Writes values to specific positions in the input tensor specified by a 1D indices array.
/// Positions not specified by indices are copied from the input tensor unchanged.
///
/// # Arguments
/// * `kernel_name` - The index_put kernel to execute (e.g., index_put::F32)
/// * `input` - Pointer to input tensor data
/// * `indices` - Pointer to int32 indices array
/// * `values` - Pointer to values tensor to write
/// * `output` - Pointer to output tensor buffer
/// * `metadata` - Tensor metadata array (see layout below)
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements, same as input)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape (shape of input/output tensor)
/// - metadata[2+num_dims..2+2*num_dims]: input_strides (strides of input tensor)
/// - metadata[2+2*num_dims..2+3*num_dims]: values_strides (strides of values tensor)
/// - metadata[2+3*num_dims]: input_offset (starting offset in input tensor)
/// - metadata[2+3*num_dims+1]: values_offset (starting offset in values tensor)
/// - metadata[2+3*num_dims+2]: dim (dimension along which to write)
/// - metadata[2+3*num_dims+3]: num_indices (number of indices)
///
/// # Safety
/// This function uses unsafe FFI calls to C kernels. Caller must ensure:
/// - All pointers are valid and properly aligned
/// - Metadata accurately describes tensor layout
/// - Output buffer has sufficient capacity
///
/// # Returns
/// Returns `Ok(())` on success.
pub fn call_ops_index_put(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    indices: *const i32,
    values: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_index_put(kernel_name.0, input, indices, values, output, metadata.as_ptr());
    }

    Ok(())
}

/// Execute a gather operation
///
/// Gathers elements from the input tensor along a specified dimension using an indices tensor.
/// The indices tensor can have any shape, and the output will have the same shape with the
/// gather dimension replaced by the indices shape.
///
/// # Arguments
/// * `kernel_name` - The gather kernel to execute (e.g., gather::F32)
/// * `input` - Pointer to input tensor data
/// * `indices` - Pointer to int32 indices tensor
/// * `output` - Pointer to output tensor buffer
/// * `metadata` - Tensor metadata array (see layout below)
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape (shape of input tensor)
/// - metadata[2+num_dims..2+2*num_dims]: input_strides (strides of input tensor)
/// - metadata[2+2*num_dims..2+3*num_dims]: indices_strides (strides of indices tensor)
/// - metadata[2+3*num_dims]: input_offset (starting offset in input tensor)
/// - metadata[2+3*num_dims+1]: indices_offset (starting offset in indices tensor)
/// - metadata[2+3*num_dims+2]: dim (dimension along which to gather)
/// - metadata[2+3*num_dims+3]: indices_len (total number of indices)
///
/// # Safety
/// This function uses unsafe FFI calls to C kernels. Caller must ensure:
/// - All pointers are valid and properly aligned
/// - Metadata accurately describes tensor layout
/// - Output buffer has sufficient capacity
///
/// # Returns
/// Returns `Ok(())` on success.
pub fn call_ops_gather(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    indices: *const i32,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        dispatch_gather(kernel_name.0, input, indices, output, metadata.as_ptr());
    }

    Ok(())
}

/// Execute a scatter operation (scatter, scatter_add, scatter_max, scatter_min)
///
/// Scatters elements from src tensor into output at positions specified by indices tensor.
/// The operation first copies input to output, then scatters src values.
///
/// Variants:
/// - `scatter`: Overwrites values at specified positions
/// - `scatter_add`: Accumulates (adds) values at specified positions
/// - `scatter_max`: Takes maximum of existing and new values
/// - `scatter_min`: Takes minimum of existing and new values
///
/// # Arguments
/// * `kernel_name` - The scatter kernel to execute (e.g., scatter::F32, scatter_add::I32)
/// * `input` - Pointer to input tensor data
/// * `indices` - Pointer to int32 indices tensor
/// * `src` - Pointer to source values tensor
/// * `output` - Pointer to output tensor buffer
/// * `metadata` - Tensor metadata array (see layout below)
///
/// # Metadata layout
/// - metadata[0]: num_els (number of elements in src tensor to scatter)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape (shape of input/output tensor)
/// - metadata[2+num_dims..2+2*num_dims]: input_strides (strides of input tensor)
/// - metadata[2+2*num_dims..2+3*num_dims]: src_shape (shape of src tensor)
/// - metadata[2+3*num_dims..2+4*num_dims]: src_strides (strides of src tensor)
/// - metadata[2+4*num_dims..2+5*num_dims]: indices_strides (strides of indices tensor)
/// - metadata[2+5*num_dims]: input_offset (starting offset in input tensor)
/// - metadata[2+5*num_dims+1]: src_offset (starting offset in src tensor)
/// - metadata[2+5*num_dims+2]: indices_offset (starting offset in indices tensor)
/// - metadata[2+5*num_dims+3]: dim (dimension along which to scatter)
///
/// # Safety
/// This function uses unsafe FFI calls to C kernels. Caller must ensure:
/// - All pointers are valid and properly aligned
/// - Metadata accurately describes tensor layout
/// - Output buffer has sufficient capacity (same as input)
/// - src and indices tensors have compatible shapes
///
/// # Returns
/// Returns `Ok(())` on success.
pub fn call_ops_scatter(
    kernel_name: crate::kernels::macros::Kernel,
    input: *const c_void,
    indices: *const i32,
    src: *const c_void,
    output: *mut c_void,
    metadata: &[usize],
) -> Result<()> {
    unsafe {
        let name = kernel_name.0;
        if name.starts_with("hodu_cpu_scatter_add_") {
            dispatch_scatter_add(name, input, indices, src, output, metadata.as_ptr());
        } else if name.starts_with("hodu_cpu_scatter_max_") {
            dispatch_scatter_max(name, input, indices, src, output, metadata.as_ptr());
        } else if name.starts_with("hodu_cpu_scatter_min_") {
            dispatch_scatter_min(name, input, indices, src, output, metadata.as_ptr());
        } else {
            dispatch_scatter(name, input, indices, src, output, metadata.as_ptr());
        }
    }

    Ok(())
}
