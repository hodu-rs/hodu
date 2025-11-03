use hodu_cpu_kernels::*;

fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t * b) / b).collect()
}

#[test]
fn test_matmul_f32_2d() {
    // Matrix A: 2x3 = [[1, 2, 3], [4, 5, 6]]
    let lhs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // Matrix B: 3x2 = [[7, 8], [9, 10], [11, 12]]
    let rhs = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];

    let m = 2;
    let k = 3;
    let n = 2;

    let lhs_ndim = 2;
    let rhs_ndim = 2;
    let batch_ndim = 0;

    let mut output = vec![0.0f32; m * n];
    let num_els = m * n;

    // Metadata for matmul:
    // [num_els, lhs_ndim, rhs_ndim, batch_ndim, lhs_shape, rhs_shape, batch_shape (empty),
    //  lhs_strides, rhs_strides, lhs_offset, rhs_offset, M, K, N]
    let lhs_shape = vec![m, k];
    let rhs_shape = vec![k, n];
    let batch_shape: Vec<usize> = vec![];
    let lhs_strides = vec![3, 1]; // row-major
    let rhs_strides = vec![2, 1]; // row-major
    let lhs_offset = 0;
    let rhs_offset = 0;

    let mut metadata = vec![num_els, lhs_ndim, rhs_ndim, batch_ndim];
    metadata.extend(&lhs_shape);
    metadata.extend(&rhs_shape);
    metadata.extend(&batch_shape);
    metadata.extend(&lhs_strides);
    metadata.extend(&rhs_strides);
    metadata.push(lhs_offset);
    metadata.push(rhs_offset);
    metadata.push(m);
    metadata.push(k);
    metadata.push(n);

    call_matmul(
        matmul::F32,
        lhs.as_ptr() as *const core::ffi::c_void,
        rhs.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //         = [[58, 64], [139, 154]]
    assert_eq!(approx(output, 4), vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_matmul_f32_batch() {
    // Batch of 2 matrices:
    // Batch 0: A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // Batch 1: A = [[2, 3], [4, 5]], B = [[6, 7], [8, 9]]
    let lhs = [1.0f32, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0];
    let rhs = [5.0f32, 6.0, 7.0, 8.0, 6.0, 7.0, 8.0, 9.0];

    let batch_size = 2;
    let m = 2;
    let k = 2;
    let n = 2;

    let lhs_ndim = 3; // [batch, m, k]
    let rhs_ndim = 3; // [batch, k, n]
    let batch_ndim = 1;

    let mut output = vec![0.0f32; batch_size * m * n];
    let num_els = batch_size * m * n;

    // Metadata for batched matmul:
    let lhs_shape = vec![batch_size, m, k];
    let rhs_shape = vec![batch_size, k, n];
    let batch_shape = vec![batch_size];
    let lhs_strides = vec![4, 2, 1]; // batch stride = 4, row stride = 2, col stride = 1
    let rhs_strides = vec![4, 2, 1];
    let lhs_offset = 0;
    let rhs_offset = 0;

    let mut metadata = vec![num_els, lhs_ndim, rhs_ndim, batch_ndim];
    metadata.extend(&lhs_shape);
    metadata.extend(&rhs_shape);
    metadata.extend(&batch_shape);
    metadata.extend(&lhs_strides);
    metadata.extend(&rhs_strides);
    metadata.push(lhs_offset);
    metadata.push(rhs_offset);
    metadata.push(m);
    metadata.push(k);
    metadata.push(n);

    call_matmul(
        matmul::F32,
        lhs.as_ptr() as *const core::ffi::c_void,
        rhs.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Expected:
    // Batch 0: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    // Batch 1: [[2*6+3*8, 2*7+3*9], [4*6+5*8, 4*7+5*9]] = [[36, 41], [64, 73]]
    assert_eq!(approx(output, 4), vec![19.0, 22.0, 43.0, 50.0, 36.0, 41.0, 64.0, 73.0]);
}

#[test]
fn test_dot_f32_simple() {
    // Matrix A: 2x3 = [[1, 2, 3], [4, 5, 6]]
    let lhs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // Matrix B: 3x2 = [[7, 8], [9, 10], [11, 12]]
    let rhs = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];

    let m = 2; // rows of A
    let k = 3; // cols of A / rows of B
    let n = 2; // cols of B

    let mut output = vec![0.0f32; m * n];

    // Metadata: [M, K, N, lhs_stride_m, lhs_stride_k, rhs_stride_k, rhs_stride_n, lhs_offset, rhs_offset]
    let metadata = vec![m, k, n, 3, 1, 2, 1, 0, 0];

    call_dot(
        dot::F32,
        lhs.as_ptr() as *const core::ffi::c_void,
        rhs.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //         = [[58, 64], [139, 154]]
    assert_eq!(approx(output, 4), vec![58.0, 64.0, 139.0, 154.0]);
}
