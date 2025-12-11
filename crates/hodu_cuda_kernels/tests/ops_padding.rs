use hodu_cuda_kernels::{kernel::Kernels, kernels::*};
use std::sync::Arc;

fn device() -> Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

fn kernels() -> Kernels {
    Kernels::new()
}

#[test]
fn pad_constant_f32_1d() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1.0f32, 2.0, 3.0];
    let pad_value = vec![0.0f32];
    let input_dev = stream.memcpy_stod(&input).unwrap();
    let pad_value_dev = stream.memcpy_stod(&pad_value).unwrap();
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(6).unwrap() };

    let num_dims = 1;
    let input_shape = vec![3];
    let output_shape = vec![6];
    let pad_before = vec![2];
    let num_els = 6;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_constant(
        pad_constant::F32,
        &kernels,
        &device,
        &input_dev,
        &mut output,
        &pad_value_dev,
        &metadata,
    )
    .unwrap();

    let mut results = vec![0.0f32; 6];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
}

#[test]
fn pad_constant_f32_2d() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let pad_value = vec![0.0f32];
    let input_dev = stream.memcpy_stod(&input).unwrap();
    let pad_value_dev = stream.memcpy_stod(&pad_value).unwrap();
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(16).unwrap() };

    let num_dims = 2;
    let input_shape = vec![2, 2];
    let output_shape = vec![4, 4];
    let pad_before = vec![1, 1];
    let num_els = 16;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_constant(
        pad_constant::F32,
        &kernels,
        &device,
        &input_dev,
        &mut output,
        &pad_value_dev,
        &metadata,
    )
    .unwrap();

    let mut results = vec![0.0f32; 16];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    #[rustfmt::skip]
    let expected = vec![
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 2.0, 0.0,
        0.0, 3.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(results, expected);
}

#[test]
fn pad_reflect_f32_1d() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(8).unwrap() };

    let num_dims = 1;
    let input_shape = vec![4];
    let output_shape = vec![8];
    let pad_before = vec![2];
    let num_els = 8;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_reflect(pad_reflect::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; 8];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
}

#[test]
fn pad_replicate_f32_1d() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1.0f32, 2.0, 3.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(7).unwrap() };

    let num_dims = 1;
    let input_shape = vec![3];
    let output_shape = vec![7];
    let pad_before = vec![2];
    let num_els = 7;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_replicate(
        pad_replicate::F32,
        &kernels,
        &device,
        &input_dev,
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![0.0f32; 7];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]);
}

#[test]
fn pad_circular_f32_1d() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1.0f32, 2.0, 3.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(7).unwrap() };

    let num_dims = 1;
    let input_shape = vec![3];
    let output_shape = vec![7];
    let pad_before = vec![2];
    let num_els = 7;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_circular(pad_circular::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; 7];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, vec![2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]);
}
