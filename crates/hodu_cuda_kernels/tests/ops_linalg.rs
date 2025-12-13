use hodu_cuda_kernels::{
    cuda::CudaSlice,
    kernel::Kernels,
    kernels::{call_ops_det, call_ops_inv, det, inv},
};
use std::sync::Arc;

fn device() -> Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

fn kernels() -> Kernels {
    Kernels::new()
}

fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t * b) / b).collect()
}

#[test]
fn det_f32_2x2() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    // 2x2 matrix: [[1, 2], [3, 4]]
    // det = 1*4 - 2*3 = -2
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let mut output: CudaSlice<f32> = unsafe { stream.alloc(1).unwrap() };

    // Metadata: [batch_size, n, ndim, shape..., strides..., offset]
    let metadata = vec![1, 2, 2, 2, 2, 2, 1, 0];

    call_ops_det(det::F32, &kernels, &device, &input_dev, &mut output, 1, &metadata).unwrap();

    let mut results = vec![0.0f32; 1];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(approx(results, 4), vec![-2.0]);
}

#[test]
fn det_f32_3x3() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    // 3x3 matrix: [[6, 1, 1], [4, -2, 5], [2, 8, 7]]
    // det = -306
    let input = vec![6.0f32, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let mut output: CudaSlice<f32> = unsafe { stream.alloc(1).unwrap() };

    let metadata = vec![1, 3, 2, 3, 3, 3, 1, 0];

    call_ops_det(det::F32, &kernels, &device, &input_dev, &mut output, 1, &metadata).unwrap();

    let mut results = vec![0.0f32; 1];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(approx(results, 2), vec![-306.0]);
}

#[test]
fn det_f32_4x4_identity() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    // 4x4 identity matrix: det(I) = 1
    let input = vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let mut output: CudaSlice<f32> = unsafe { stream.alloc(1).unwrap() };

    let metadata = vec![1, 4, 2, 4, 4, 4, 1, 0];

    call_ops_det(det::F32, &kernels, &device, &input_dev, &mut output, 1, &metadata).unwrap();

    let mut results = vec![0.0f32; 1];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(approx(results, 4), vec![1.0]);
}

#[test]
fn det_f32_batch() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    // Batch of 2 matrices:
    // Matrix 0: [[1, 2], [3, 4]], det = -2
    // Matrix 1: [[5, 6], [7, 8]], det = -2
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let mut output: CudaSlice<f32> = unsafe { stream.alloc(2).unwrap() };

    // Metadata: [batch_size, n, ndim, shape..., strides..., offset]
    // shape = [2, 2, 2], strides = [4, 2, 1], offset = 0
    let metadata = vec![2, 2, 3, 2, 2, 2, 4, 2, 1, 0];

    call_ops_det(det::F32, &kernels, &device, &input_dev, &mut output, 2, &metadata).unwrap();

    let mut results = vec![0.0f32; 2];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(approx(results, 4), vec![-2.0, -2.0]);
}

// ============================================================================
// INV (Matrix Inverse) Tests
// ============================================================================

#[test]
fn inv_f32_1x1() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    // 1x1 matrix: inv([[5]]) = [[0.2]]
    let input = vec![5.0f32];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let mut output: CudaSlice<f32> = unsafe { stream.alloc(1).unwrap() };

    let metadata = vec![1, 1, 2, 1, 1, 1, 1, 0];

    call_ops_inv(inv::F32, &kernels, &device, &input_dev, &mut output, 1, &metadata).unwrap();

    let mut results = vec![0.0f32; 1];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(approx(results, 4), vec![0.2]);
}

#[test]
fn inv_f32_2x2() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    // 2x2 matrix: [[4, 7], [2, 6]]
    // inv = [[0.6, -0.7], [-0.2, 0.4]]
    let input = vec![4.0f32, 7.0, 2.0, 6.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let mut output: CudaSlice<f32> = unsafe { stream.alloc(4).unwrap() };

    let metadata = vec![1, 2, 2, 2, 2, 2, 1, 0];

    call_ops_inv(inv::F32, &kernels, &device, &input_dev, &mut output, 1, &metadata).unwrap();

    let mut results = vec![0.0f32; 4];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(approx(results, 4), vec![0.6, -0.7, -0.2, 0.4]);
}

#[test]
fn inv_f32_4x4_identity() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    // 4x4 identity matrix: inv(I) = I
    let input = vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let mut output: CudaSlice<f32> = unsafe { stream.alloc(16).unwrap() };

    let metadata = vec![1, 4, 2, 4, 4, 4, 1, 0];

    call_ops_inv(inv::F32, &kernels, &device, &input_dev, &mut output, 1, &metadata).unwrap();

    let mut results = vec![0.0f32; 16];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(approx(results, 4), input);
}

#[test]
fn inv_f32_batch() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    // Batch of 2 matrices:
    // Matrix 0: [[2, 0], [0, 2]], inv = [[0.5, 0], [0, 0.5]]
    // Matrix 1: [[1, 1], [0, 1]], inv = [[1, -1], [0, 1]]
    let input = vec![2.0f32, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 1.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let mut output: CudaSlice<f32> = unsafe { stream.alloc(8).unwrap() };

    let metadata = vec![2, 2, 3, 2, 2, 2, 4, 2, 1, 0];

    call_ops_inv(inv::F32, &kernels, &device, &input_dev, &mut output, 2, &metadata).unwrap();

    let mut results = vec![0.0f32; 8];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    let expected = vec![0.5, 0.0, 0.0, 0.5, 1.0, -1.0, 0.0, 1.0];
    assert_eq!(approx(results, 4), expected);
}
