use crate::kernels::{macros::ops, Kernel};
use std::ffi::c_void;

ops!(
    reduce_window_max,
    reduce_window_mean,
    reduce_window_sum,
    reduce_window_min
);

extern "C" {
    fn reduce_window_max_f8e4m3(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_max_f8e5m2(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_max_bf16(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_max_f16(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_max_f32(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_max_f64(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_max_i8(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_max_i16(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_max_i32(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_max_i64(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_max_u8(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_max_u16(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_max_u32(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_max_u64(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );

    fn reduce_window_mean_f8e4m3(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_mean_f8e5m2(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_mean_bf16(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_mean_f16(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_mean_f32(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_mean_f64(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );

    fn reduce_window_sum_f8e4m3(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_sum_f8e5m2(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_sum_bf16(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_sum_f16(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_sum_f32(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_sum_f64(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_sum_i8(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_sum_i16(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_sum_i32(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_sum_i64(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_sum_u8(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_sum_u16(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_sum_u32(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_sum_u64(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );

    fn reduce_window_min_f8e4m3(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_min_f8e5m2(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_min_bf16(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_min_f16(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_min_f32(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_min_f64(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_min_i8(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_min_i16(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_min_i32(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_min_i64(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_min_u8(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_min_u16(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_min_u32(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
    fn reduce_window_min_u64(
        input: *const c_void,
        output: *mut c_void,
        num_els: usize,
        num_dims: usize,
        metadata: *const usize,
    );
}

pub fn call_reduce_window(
    kernel_name: Kernel,
    input: *const c_void,
    output: *mut c_void,
    num_els: usize,
    num_dims: usize,
    metadata: &[usize],
) {
    let kernel_str = kernel_name.0;
    unsafe {
        match kernel_str {
            "reduce_window_max_f8e4m3" => reduce_window_max_f8e4m3(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_max_f8e5m2" => reduce_window_max_f8e5m2(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_max_bf16" => reduce_window_max_bf16(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_max_f16" => reduce_window_max_f16(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_max_f32" => reduce_window_max_f32(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_max_f64" => reduce_window_max_f64(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_max_i8" => reduce_window_max_i8(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_max_i16" => reduce_window_max_i16(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_max_i32" => reduce_window_max_i32(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_max_i64" => reduce_window_max_i64(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_max_u8" => reduce_window_max_u8(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_max_u16" => reduce_window_max_u16(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_max_u32" => reduce_window_max_u32(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_max_u64" => reduce_window_max_u64(input, output, num_els, num_dims, metadata.as_ptr()),

            "reduce_window_mean_f8e4m3" => {
                reduce_window_mean_f8e4m3(input, output, num_els, num_dims, metadata.as_ptr())
            },
            "reduce_window_mean_f8e5m2" => {
                reduce_window_mean_f8e5m2(input, output, num_els, num_dims, metadata.as_ptr())
            },
            "reduce_window_mean_bf16" => reduce_window_mean_bf16(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_mean_f16" => reduce_window_mean_f16(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_mean_f32" => reduce_window_mean_f32(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_mean_f64" => reduce_window_mean_f64(input, output, num_els, num_dims, metadata.as_ptr()),

            "reduce_window_sum_f8e4m3" => reduce_window_sum_f8e4m3(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_sum_f8e5m2" => reduce_window_sum_f8e5m2(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_sum_bf16" => reduce_window_sum_bf16(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_sum_f16" => reduce_window_sum_f16(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_sum_f32" => reduce_window_sum_f32(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_sum_f64" => reduce_window_sum_f64(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_sum_i8" => reduce_window_sum_i8(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_sum_i16" => reduce_window_sum_i16(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_sum_i32" => reduce_window_sum_i32(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_sum_i64" => reduce_window_sum_i64(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_sum_u8" => reduce_window_sum_u8(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_sum_u16" => reduce_window_sum_u16(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_sum_u32" => reduce_window_sum_u32(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_sum_u64" => reduce_window_sum_u64(input, output, num_els, num_dims, metadata.as_ptr()),

            "reduce_window_min_f8e4m3" => reduce_window_min_f8e4m3(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_min_f8e5m2" => reduce_window_min_f8e5m2(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_min_bf16" => reduce_window_min_bf16(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_min_f16" => reduce_window_min_f16(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_min_f32" => reduce_window_min_f32(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_min_f64" => reduce_window_min_f64(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_min_i8" => reduce_window_min_i8(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_min_i16" => reduce_window_min_i16(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_min_i32" => reduce_window_min_i32(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_min_i64" => reduce_window_min_i64(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_min_u8" => reduce_window_min_u8(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_min_u16" => reduce_window_min_u16(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_min_u32" => reduce_window_min_u32(input, output, num_els, num_dims, metadata.as_ptr()),
            "reduce_window_min_u64" => reduce_window_min_u64(input, output, num_els, num_dims, metadata.as_ptr()),

            _ => panic!("Unsupported reduce_window kernel: {:?}", kernel_name),
        }
    }
}
