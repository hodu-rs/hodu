use hodu_cuda_kernels::{cuda::CudaSlice, kernel::Kernels, kernels::*};
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
fn dot_f32_simple() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    // Matrix A: 2x3 = [[1, 2, 3], [4, 5, 6]]
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // Matrix B: 3x2 = [[7, 8], [9, 10], [11, 12]]
    let rhs = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];

    let lhs_dev = stream.memcpy_stod(&lhs).unwrap();
    let rhs_dev = stream.memcpy_stod(&rhs).unwrap();

    let m = 2; // rows of A
    let k = 3; // cols of A / rows of B
    let n = 2; // cols of B

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(m * n).unwrap() };

    // Metadata layout (9 elements):
    // [M, K, N, lhs_stride_m, lhs_stride_k, rhs_stride_k, rhs_stride_n, lhs_offset, rhs_offset]
    let metadata = vec![m, k, n, 3, 1, 2, 1, 0, 0];

    call_ops_dot(dot::F32, &kernels, &device, &lhs_dev, &rhs_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; m * n];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //         = [[58, 64], [139, 154]]
    assert_eq!(approx(results, 4), vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_f32_2d() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    // Matrix A: 2x3 = [[1, 2, 3], [4, 5, 6]]
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // Matrix B: 3x2 = [[7, 8], [9, 10], [11, 12]]
    let rhs = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];

    let lhs_dev = stream.memcpy_stod(&lhs).unwrap();
    let rhs_dev = stream.memcpy_stod(&rhs).unwrap();

    let m = 2;
    let k = 3;
    let n = 2;
    let num_els = m * n;

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(num_els).unwrap() };

    // Metadata for matmul: [num_els, lhs_ndim, rhs_ndim, batch_ndim, lhs_shape, rhs_shape,
    //                       lhs_strides, rhs_strides, lhs_offset, rhs_offset, M, K, N]
    let lhs_ndim = 2;
    let rhs_ndim = 2;
    let batch_ndim = 0;
    let metadata = vec![
        num_els, lhs_ndim, rhs_ndim, batch_ndim, m, k, // lhs shape
        k, n, // rhs shape
        k, 1, // lhs strides
        n, 1, // rhs strides
        0, 0, // offsets
        m, k, n, // M, K, N
    ];

    call_ops_matmul(
        matmul::F32,
        &kernels,
        &device,
        &lhs_dev,
        &rhs_dev,
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![0.0f32; num_els];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(approx(results, 4), vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_f32_square() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    // Square matrices: 3x3
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let rhs = vec![9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    let lhs_dev = stream.memcpy_stod(&lhs).unwrap();
    let rhs_dev = stream.memcpy_stod(&rhs).unwrap();

    let m = 3;
    let k = 3;
    let n = 3;
    let num_els = m * n;

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(num_els).unwrap() };

    let lhs_ndim = 2;
    let rhs_ndim = 2;
    let batch_ndim = 0;
    let metadata = vec![
        num_els, lhs_ndim, rhs_ndim, batch_ndim, m, k, k, n, k, 1, n, 1, 0, 0, m, k, n,
    ];

    call_ops_matmul(
        matmul::F32,
        &kernels,
        &device,
        &lhs_dev,
        &rhs_dev,
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![0.0f32; num_els];
    stream.memcpy_dtoh(&output, &mut results).unwrap();

    // Expected result calculation:
    // Row 0: [1*9+2*6+3*3, 1*8+2*5+3*2, 1*7+2*4+3*1] = [30, 24, 18]
    // Row 1: [4*9+5*6+6*3, 4*8+5*5+6*2, 4*7+5*4+6*1] = [84, 69, 54]
    // Row 2: [7*9+8*6+9*3, 7*8+8*5+9*2, 7*7+8*4+9*1] = [138, 114, 90]
    let expected = vec![30.0, 24.0, 18.0, 84.0, 69.0, 54.0, 138.0, 114.0, 90.0];
    assert_eq!(approx(results, 4), expected);
}

#[test]
fn dot_f32_identity() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    // 3x3 identity matrix multiplied by itself
    let identity = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let identity_dev = stream.memcpy_stod(&identity).unwrap();

    let m = 3;
    let k = 3;
    let n = 3;

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(m * n).unwrap() };

    let metadata = vec![m, k, n, 3, 1, 3, 1, 0, 0];

    call_ops_dot(
        dot::F32,
        &kernels,
        &device,
        &identity_dev,
        &identity_dev,
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![0.0f32; m * n];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(results, identity);
}

#[test]
fn matmul_f32_batch() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    // Batch of 2 matrices: 2x3 @ 3x2
    // Batch 0: [[1, 2, 3], [4, 5, 6]]
    // Batch 1: [[7, 8, 9], [10, 11, 12]]
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    // Weight (shared): [[1, 0], [0, 1], [1, 1]]
    let rhs = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];

    let lhs_dev = stream.memcpy_stod(&lhs).unwrap();
    let rhs_dev = stream.memcpy_stod(&rhs).unwrap();

    let batch = 2;
    let m = 2;
    let k = 3;
    let n = 2;
    let num_els = batch * m * n;

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(num_els).unwrap() };

    // Metadata for batched matmul: [num_els, lhs_ndim, rhs_ndim, batch_ndim, lhs_shape, rhs_shape,
    //                                batch_shape, lhs_strides, rhs_strides, lhs_offset, rhs_offset, M, K, N]
    let lhs_ndim = 3;
    let rhs_ndim = 2;
    let batch_ndim = 1;
    let metadata = vec![
        num_els,
        lhs_ndim,
        rhs_ndim,
        batch_ndim,
        batch,
        m,
        k, // lhs shape [batch, m, k]
        k,
        n,     // rhs shape [k, n]
        batch, // batch_shape [batch]
        m * k,
        k,
        1, // lhs strides
        n,
        1, // rhs strides
        0,
        0, // offsets
        m,
        k,
        n, // M, K, N
    ];

    call_ops_matmul(
        matmul::F32,
        &kernels,
        &device,
        &lhs_dev,
        &rhs_dev,
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![0.0f32; num_els];
    stream.memcpy_dtoh(&output, &mut results).unwrap();

    // Batch 0: [[1, 2, 3], [4, 5, 6]] @ [[1, 0], [0, 1], [1, 1]]
    //        = [[1*1+2*0+3*1, 1*0+2*1+3*1], [4*1+5*0+6*1, 4*0+5*1+6*1]]
    //        = [[4, 5], [10, 11]]
    // Batch 1: [[7, 8, 9], [10, 11, 12]] @ [[1, 0], [0, 1], [1, 1]]
    //        = [[7*1+8*0+9*1, 7*0+8*1+9*1], [10*1+11*0+12*1, 10*0+11*1+12*1]]
    //        = [[16, 17], [22, 23]]
    assert_eq!(approx(results, 4), vec![4.0, 5.0, 10.0, 11.0, 16.0, 17.0, 22.0, 23.0]);
}
