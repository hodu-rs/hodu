use crate::kernels::macros::{ops, Kernel};
use std::ffi::c_void;

ops!(
    conv1d,
    conv2d,
    conv3d,
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
    conv1d_grad_weight,
    conv2d_grad_weight,
    conv3d_grad_weight,
    conv_transpose1d_grad_weight,
    conv_transpose2d_grad_weight,
    conv_transpose3d_grad_weight
);

extern "C" {
    fn conv1d_f8e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv1d_f8e5m2(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv1d_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv1d_f16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv1d_f32(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv1d_f64(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );

    fn conv2d_f8e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv2d_f8e5m2(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv2d_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv2d_f16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv2d_f32(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv2d_f64(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );

    fn conv3d_f8e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv3d_f8e5m2(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv3d_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv3d_f16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv3d_f32(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv3d_f64(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );

    fn conv_transpose1d_f8e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose1d_f8e5m2(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose1d_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose1d_f16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose1d_f32(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose1d_f64(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );

    fn conv_transpose2d_f8e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose2d_f8e5m2(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose2d_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose2d_f16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose2d_f32(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose2d_f64(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );

    fn conv_transpose3d_f8e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose3d_f8e5m2(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose3d_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose3d_f16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose3d_f32(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose3d_f64(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );

    fn conv1d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv1d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv1d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv1d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv1d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv1d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );

    fn conv2d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv2d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv2d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv2d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv2d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv2d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );

    fn conv3d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv3d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv3d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv3d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv3d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv3d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );

    fn conv_transpose1d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose1d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose1d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose1d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose1d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose1d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );

    fn conv_transpose2d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose2d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose2d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose2d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose2d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose2d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );

    fn conv_transpose3d_grad_weight_f8e4m3(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose3d_grad_weight_f8e5m2(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose3d_grad_weight_bf16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose3d_grad_weight_f16(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose3d_grad_weight_f32(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
    fn conv_transpose3d_grad_weight_f64(
        input: *const c_void,
        grad_output: *const c_void,
        grad_weight: *mut c_void,
        num_els: usize,
        metadata: *const usize,
    );
}

pub fn call_conv(
    kernel: Kernel,
    input: *const c_void,
    weight: *const c_void,
    output: *mut c_void,
    num_els: usize,
    metadata: &[usize],
) {
    unsafe {
        match kernel {
            conv1d::F8E4M3 => conv1d_f8e4m3(input, weight, output, num_els, metadata.as_ptr()),
            conv1d::F8E5M2 => conv1d_f8e5m2(input, weight, output, num_els, metadata.as_ptr()),
            conv1d::BF16 => conv1d_bf16(input, weight, output, num_els, metadata.as_ptr()),
            conv1d::F16 => conv1d_f16(input, weight, output, num_els, metadata.as_ptr()),
            conv1d::F32 => conv1d_f32(input, weight, output, num_els, metadata.as_ptr()),
            conv1d::F64 => conv1d_f64(input, weight, output, num_els, metadata.as_ptr()),
            conv2d::F8E4M3 => conv2d_f8e4m3(input, weight, output, num_els, metadata.as_ptr()),
            conv2d::F8E5M2 => conv2d_f8e5m2(input, weight, output, num_els, metadata.as_ptr()),
            conv2d::BF16 => conv2d_bf16(input, weight, output, num_els, metadata.as_ptr()),
            conv2d::F16 => conv2d_f16(input, weight, output, num_els, metadata.as_ptr()),
            conv2d::F32 => conv2d_f32(input, weight, output, num_els, metadata.as_ptr()),
            conv2d::F64 => conv2d_f64(input, weight, output, num_els, metadata.as_ptr()),
            conv3d::F8E4M3 => conv3d_f8e4m3(input, weight, output, num_els, metadata.as_ptr()),
            conv3d::F8E5M2 => conv3d_f8e5m2(input, weight, output, num_els, metadata.as_ptr()),
            conv3d::BF16 => conv3d_bf16(input, weight, output, num_els, metadata.as_ptr()),
            conv3d::F16 => conv3d_f16(input, weight, output, num_els, metadata.as_ptr()),
            conv3d::F32 => conv3d_f32(input, weight, output, num_els, metadata.as_ptr()),
            conv3d::F64 => conv3d_f64(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose1d::F8E4M3 => conv_transpose1d_f8e4m3(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose1d::F8E5M2 => conv_transpose1d_f8e5m2(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose1d::BF16 => conv_transpose1d_bf16(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose1d::F16 => conv_transpose1d_f16(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose1d::F32 => conv_transpose1d_f32(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose1d::F64 => conv_transpose1d_f64(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose2d::F8E4M3 => conv_transpose2d_f8e4m3(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose2d::F8E5M2 => conv_transpose2d_f8e5m2(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose2d::BF16 => conv_transpose2d_bf16(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose2d::F16 => conv_transpose2d_f16(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose2d::F32 => conv_transpose2d_f32(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose2d::F64 => conv_transpose2d_f64(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose3d::F8E4M3 => conv_transpose3d_f8e4m3(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose3d::F8E5M2 => conv_transpose3d_f8e5m2(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose3d::BF16 => conv_transpose3d_bf16(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose3d::F16 => conv_transpose3d_f16(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose3d::F32 => conv_transpose3d_f32(input, weight, output, num_els, metadata.as_ptr()),
            conv_transpose3d::F64 => conv_transpose3d_f64(input, weight, output, num_els, metadata.as_ptr()),
            _ => panic!("Unsupported conv kernel: {:?}", kernel),
        }
    }
}

pub fn call_conv_grad_weight(
    kernel: Kernel,
    input: *const c_void,
    grad_output: *const c_void,
    grad_weight: *mut c_void,
    num_els: usize,
    metadata: &[usize],
) {
    unsafe {
        match kernel {
            conv1d_grad_weight::F8E4M3 => {
                conv1d_grad_weight_f8e4m3(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv1d_grad_weight::F8E5M2 => {
                conv1d_grad_weight_f8e5m2(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv1d_grad_weight::BF16 => {
                conv1d_grad_weight_bf16(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv1d_grad_weight::F16 => {
                conv1d_grad_weight_f16(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv1d_grad_weight::F32 => {
                conv1d_grad_weight_f32(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv1d_grad_weight::F64 => {
                conv1d_grad_weight_f64(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv2d_grad_weight::F8E4M3 => {
                conv2d_grad_weight_f8e4m3(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv2d_grad_weight::F8E5M2 => {
                conv2d_grad_weight_f8e5m2(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv2d_grad_weight::BF16 => {
                conv2d_grad_weight_bf16(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv2d_grad_weight::F16 => {
                conv2d_grad_weight_f16(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv2d_grad_weight::F32 => {
                conv2d_grad_weight_f32(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv2d_grad_weight::F64 => {
                conv2d_grad_weight_f64(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv3d_grad_weight::F8E4M3 => {
                conv3d_grad_weight_f8e4m3(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv3d_grad_weight::F8E5M2 => {
                conv3d_grad_weight_f8e5m2(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv3d_grad_weight::BF16 => {
                conv3d_grad_weight_bf16(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv3d_grad_weight::F16 => {
                conv3d_grad_weight_f16(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv3d_grad_weight::F32 => {
                conv3d_grad_weight_f32(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv3d_grad_weight::F64 => {
                conv3d_grad_weight_f64(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::F8E4M3 => {
                conv_transpose1d_grad_weight_f8e4m3(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::F8E5M2 => {
                conv_transpose1d_grad_weight_f8e5m2(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::BF16 => {
                conv_transpose1d_grad_weight_bf16(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::F16 => {
                conv_transpose1d_grad_weight_f16(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::F32 => {
                conv_transpose1d_grad_weight_f32(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose1d_grad_weight::F64 => {
                conv_transpose1d_grad_weight_f64(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::F8E4M3 => {
                conv_transpose2d_grad_weight_f8e4m3(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::F8E5M2 => {
                conv_transpose2d_grad_weight_f8e5m2(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::BF16 => {
                conv_transpose2d_grad_weight_bf16(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::F16 => {
                conv_transpose2d_grad_weight_f16(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::F32 => {
                conv_transpose2d_grad_weight_f32(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose2d_grad_weight::F64 => {
                conv_transpose2d_grad_weight_f64(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::F8E4M3 => {
                conv_transpose3d_grad_weight_f8e4m3(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::F8E5M2 => {
                conv_transpose3d_grad_weight_f8e5m2(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::BF16 => {
                conv_transpose3d_grad_weight_bf16(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::F16 => {
                conv_transpose3d_grad_weight_f16(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::F32 => {
                conv_transpose3d_grad_weight_f32(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            conv_transpose3d_grad_weight::F64 => {
                conv_transpose3d_grad_weight_f64(input, grad_output, grad_weight, num_els, metadata.as_ptr())
            },
            _ => panic!("Unsupported conv grad_weight kernel: {:?}", kernel),
        }
    }
}
