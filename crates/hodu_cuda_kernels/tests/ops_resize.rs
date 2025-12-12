use hodu_cuda_kernels::{kernel::Kernels, kernels::*};
use std::sync::Arc;

fn device() -> Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

fn kernels() -> Kernels {
    Kernels::new()
}

fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// Resize modes
const RESIZE_MODE_NEAREST: usize = 0;
const RESIZE_MODE_LINEAR: usize = 1;
const RESIZE_MODE_CUBIC: usize = 2;

// Coordinate transformation modes
const RESIZE_COORD_HALF_PIXEL: usize = 0;
const RESIZE_COORD_ASYMMETRIC: usize = 1;
const RESIZE_COORD_ALIGN_CORNERS: usize = 2;

// Nearest rounding modes
const RESIZE_NEAREST_FLOOR: usize = 0;

fn build_resize_metadata(
    input_shape: &[usize],
    input_strides: &[usize],
    offset: usize,
    output_shape: &[usize],
    mode: usize,
    coord_transform: usize,
    nearest_mode: usize,
) -> Vec<usize> {
    let output_size: usize = output_shape.iter().product();
    let num_dims = input_shape.len();
    let mut metadata = vec![output_size, num_dims];
    metadata.extend(input_shape);
    metadata.extend(input_strides);
    metadata.push(offset);
    metadata.extend(output_shape);
    metadata.push(mode);
    metadata.push(coord_transform);
    metadata.push(nearest_mode);
    metadata
}

#[test]
fn test_resize_nearest_upsample_2x_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_shape = vec![1, 1, 2, 2];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![1, 1, 4, 4];
    let output_size: usize = output_shape.iter().product();

    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output_dev: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(output_size).unwrap() };

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_NEAREST,
        RESIZE_COORD_ASYMMETRIC,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(resize::F32, &kernels, &device, &input_dev, &mut output_dev, &metadata).unwrap();

    let mut results = vec![0.0f32; output_size];
    stream.memcpy_dtoh(&output_dev, &mut results).unwrap();

    let expected = vec![
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    assert_eq!(results, expected);
}

#[test]
fn test_resize_bilinear_upsample_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_shape = vec![1, 1, 2, 2];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![1, 1, 4, 4];
    let output_size: usize = output_shape.iter().product();

    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output_dev: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(output_size).unwrap() };

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_LINEAR,
        RESIZE_COORD_ALIGN_CORNERS,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(resize::F32, &kernels, &device, &input_dev, &mut output_dev, &metadata).unwrap();

    let mut results = vec![0.0f32; output_size];
    stream.memcpy_dtoh(&output_dev, &mut results).unwrap();

    // With align_corners, corners should match exactly
    assert!((results[0] - 1.0).abs() < 1e-5); // top-left
    assert!((results[3] - 2.0).abs() < 1e-5); // top-right
    assert!((results[12] - 3.0).abs() < 1e-5); // bottom-left
    assert!((results[15] - 4.0).abs() < 1e-5); // bottom-right
}

#[test]
fn test_resize_nearest_downsample_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let input_shape = vec![1, 1, 4, 4];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![1, 1, 2, 2];
    let output_size: usize = output_shape.iter().product();

    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output_dev: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(output_size).unwrap() };

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_NEAREST,
        RESIZE_COORD_ASYMMETRIC,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(resize::F32, &kernels, &device, &input_dev, &mut output_dev, &metadata).unwrap();

    let mut results = vec![0.0f32; output_size];
    stream.memcpy_dtoh(&output_dev, &mut results).unwrap();

    assert_eq!(results, vec![1.0, 3.0, 9.0, 11.0]);
}

#[test]
fn test_resize_bilinear_half_pixel_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![0.0f32, 10.0, 10.0, 20.0];
    let input_shape = vec![1, 1, 2, 2];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![1, 1, 4, 4];
    let output_size: usize = output_shape.iter().product();

    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output_dev: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(output_size).unwrap() };

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_LINEAR,
        RESIZE_COORD_HALF_PIXEL,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(resize::F32, &kernels, &device, &input_dev, &mut output_dev, &metadata).unwrap();

    let mut results = vec![0.0f32; output_size];
    stream.memcpy_dtoh(&output_dev, &mut results).unwrap();

    // Just check output is within expected range
    for &val in &results {
        assert!(val >= 0.0 && val <= 20.0);
    }
}

#[test]
fn test_resize_bicubic_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let input_shape = vec![1, 1, 4, 4];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![1, 1, 8, 8];
    let output_size: usize = output_shape.iter().product();

    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output_dev: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(output_size).unwrap() };

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_CUBIC,
        RESIZE_COORD_HALF_PIXEL,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(resize::F32, &kernels, &device, &input_dev, &mut output_dev, &metadata).unwrap();

    let mut results = vec![0.0f32; output_size];
    stream.memcpy_dtoh(&output_dev, &mut results).unwrap();

    // Check output is within expected range (cubic can overshoot slightly)
    for &val in &results {
        assert!(val >= 0.0 && val <= 20.0);
    }
}

#[test]
fn test_resize_identity_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_shape = vec![1, 1, 2, 2];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![1, 1, 2, 2];
    let output_size: usize = output_shape.iter().product();

    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output_dev: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(output_size).unwrap() };

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_LINEAR,
        RESIZE_COORD_HALF_PIXEL,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(resize::F32, &kernels, &device, &input_dev, &mut output_dev, &metadata).unwrap();

    let mut results = vec![0.0f32; output_size];
    stream.memcpy_dtoh(&output_dev, &mut results).unwrap();

    // Identity should preserve values
    for (a, b) in results.iter().zip(input.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn test_resize_batch_channel_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let input_shape = vec![2, 2, 2, 2];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![2, 2, 4, 4];
    let output_size: usize = output_shape.iter().product();

    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output_dev: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(output_size).unwrap() };

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_NEAREST,
        RESIZE_COORD_ASYMMETRIC,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(resize::F32, &kernels, &device, &input_dev, &mut output_dev, &metadata).unwrap();

    let mut results = vec![0.0f32; output_size];
    stream.memcpy_dtoh(&output_dev, &mut results).unwrap();

    // Check that batch and channel are preserved
    assert!((results[0] - 1.0).abs() < 1e-5); // batch 0, channel 0
    assert!((results[16] - 5.0).abs() < 1e-5); // batch 0, channel 1
    assert!((results[32] - 9.0).abs() < 1e-5); // batch 1, channel 0
    assert!((results[48] - 13.0).abs() < 1e-5); // batch 1, channel 1
}

#[test]
fn test_resize_nearest_f64() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input = vec![1.0f64, 2.0, 3.0, 4.0];
    let input_shape = vec![1, 1, 2, 2];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![1, 1, 4, 4];
    let output_size: usize = output_shape.iter().product();

    let input_dev = stream.memcpy_stod(&input).unwrap();
    let mut output_dev: cudarc::driver::CudaSlice<f64> = unsafe { stream.alloc(output_size).unwrap() };

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_NEAREST,
        RESIZE_COORD_ASYMMETRIC,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(resize::F64, &kernels, &device, &input_dev, &mut output_dev, &metadata).unwrap();

    let mut results = vec![0.0f64; output_size];
    stream.memcpy_dtoh(&output_dev, &mut results).unwrap();

    let expected = vec![
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    assert_eq!(results, expected);
}
