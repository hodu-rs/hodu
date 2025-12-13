use hodu_cpu_kernels::*;

fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t * b) / b).collect()
}

fn approx_i32(v: Vec<i32>, _digits: i32) -> Vec<i32> {
    v // integers don't need approximation
}

// ============================================================================
// TRACE Tests
// ============================================================================

#[test]
fn test_trace_f32_2x2() {
    // 2x2 matrix: [[1, 2], [3, 4]]
    // trace = 1 + 4 = 5
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let mut output = [0.0f32; 1];

    let metadata = vec![1, 2, 2, 2, 2, 2, 1, 0];

    call_ops_trace(
        trace::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output.to_vec(), 4), vec![5.0]);
}

#[test]
fn test_trace_f32_3x3() {
    // 3x3 identity matrix: trace(I) = 3
    let input = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut output = [0.0f32; 1];

    let metadata = vec![1, 3, 2, 3, 3, 3, 1, 0];

    call_ops_trace(
        trace::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output.to_vec(), 4), vec![3.0]);
}

#[test]
fn test_trace_f32_batch() {
    // Batch of 2 matrices:
    // Matrix 0: [[1, 2], [3, 4]], trace = 5
    // Matrix 1: [[5, 6], [7, 8]], trace = 13
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = [0.0f32; 2];

    let metadata = vec![2, 2, 3, 2, 2, 2, 4, 2, 1, 0];

    call_ops_trace(
        trace::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output.to_vec(), 4), vec![5.0, 13.0]);
}

#[test]
fn test_trace_i32_2x2() {
    // 2x2 integer matrix: [[1, 2], [3, 4]]
    // trace = 1 + 4 = 5
    let input = [1i32, 2, 3, 4];
    let mut output = [0i32; 1];

    let metadata = vec![1, 2, 2, 2, 2, 2, 1, 0];

    call_ops_trace(
        trace::I32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output[0], 5);
}

// ============================================================================
// DET Tests
// ============================================================================

#[test]
fn test_det_f32_1x1() {
    // 1x1 matrix: det([[5]]) = 5
    let input = [5.0f32];
    let mut output = [0.0f32; 1];

    // Metadata: [batch_size, n, ndim, shape..., strides..., offset]
    let metadata = vec![1, 1, 2, 1, 1, 1, 1, 0];

    call_ops_det(
        det::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output.to_vec(), 4), vec![5.0]);
}

#[test]
fn test_det_f32_2x2() {
    // 2x2 matrix: [[1, 2], [3, 4]]
    // det = 1*4 - 2*3 = -2
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let mut output = [0.0f32; 1];

    // Metadata: [batch_size, n, ndim, shape..., strides..., offset]
    // shape = [2, 2], strides = [2, 1], offset = 0
    let metadata = vec![1, 2, 2, 2, 2, 2, 1, 0];

    call_ops_det(
        det::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output.to_vec(), 4), vec![-2.0]);
}

#[test]
fn test_det_f32_3x3() {
    // 3x3 matrix: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    // det = 1*(5*9-6*8) - 2*(4*9-6*7) + 3*(4*8-5*7)
    //     = 1*(-3) - 2*(-6) + 3*(-3) = -3 + 12 - 9 = 0
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut output = [0.0f32; 1];

    // Metadata: [batch_size, n, ndim, shape..., strides..., offset]
    // shape = [3, 3], strides = [3, 1], offset = 0
    let metadata = vec![1, 3, 2, 3, 3, 3, 1, 0];

    call_ops_det(
        det::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output.to_vec(), 4), vec![0.0]);
}

#[test]
fn test_det_f32_3x3_nonzero() {
    // 3x3 matrix: [[6, 1, 1], [4, -2, 5], [2, 8, 7]]
    // det = 6*(-2*7-5*8) - 1*(4*7-5*2) + 1*(4*8-(-2)*2)
    //     = 6*(-14-40) - 1*(28-10) + 1*(32+4)
    //     = 6*(-54) - 18 + 36 = -324 - 18 + 36 = -306
    let input = [6.0f32, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0];
    let mut output = [0.0f32; 1];

    let metadata = vec![1, 3, 2, 3, 3, 3, 1, 0];

    call_ops_det(
        det::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output.to_vec(), 2), vec![-306.0]);
}

#[test]
fn test_det_f32_4x4() {
    // 4x4 identity matrix: det(I) = 1
    let input = [
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let mut output = [0.0f32; 1];

    // Metadata: [batch_size, n, ndim, shape..., strides..., offset]
    // shape = [4, 4], strides = [4, 1], offset = 0
    let metadata = vec![1, 4, 2, 4, 4, 4, 1, 0];

    call_ops_det(
        det::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output.to_vec(), 4), vec![1.0]);
}

#[test]
fn test_det_f32_batch() {
    // Batch of 2 matrices:
    // Matrix 0: [[1, 2], [3, 4]], det = -2
    // Matrix 1: [[5, 6], [7, 8]], det = 5*8 - 6*7 = 40 - 42 = -2
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = [0.0f32; 2];

    // Metadata: [batch_size, n, ndim, shape..., strides..., offset]
    // shape = [2, 2, 2], strides = [4, 2, 1], offset = 0
    let metadata = vec![2, 2, 3, 2, 2, 2, 4, 2, 1, 0];

    call_ops_det(
        det::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output.to_vec(), 4), vec![-2.0, -2.0]);
}

#[test]
fn test_det_i32_2x2() {
    // 2x2 integer matrix: [[1, 2], [3, 4]]
    // det = 1*4 - 2*3 = -2
    let input = [1i32, 2, 3, 4];
    let mut output = [0i32; 1];

    let metadata = vec![1, 2, 2, 2, 2, 2, 1, 0];

    call_ops_det(
        det::I32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output[0], -2);
}

// ============================================================================
// INV (Matrix Inverse) Tests
// ============================================================================

#[test]
fn test_inv_f32_1x1() {
    // 1x1 matrix: inv([[5]]) = [[0.2]]
    let input = [5.0f32];
    let mut output = [0.0f32; 1];

    // Metadata: [batch_size, n, ndim, shape..., strides..., offset]
    let metadata = vec![1, 1, 2, 1, 1, 1, 1, 0];

    call_ops_inv(
        inv::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output.to_vec(), 4), vec![0.2]);
}

#[test]
fn test_inv_f32_2x2() {
    // 2x2 matrix: [[4, 7], [2, 6]]
    // det = 4*6 - 7*2 = 24 - 14 = 10
    // inv = 1/10 * [[6, -7], [-2, 4]]
    //     = [[0.6, -0.7], [-0.2, 0.4]]
    let input = [4.0f32, 7.0, 2.0, 6.0];
    let mut output = [0.0f32; 4];

    let metadata = vec![1, 2, 2, 2, 2, 2, 1, 0];

    call_ops_inv(
        inv::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output.to_vec(), 4), vec![0.6, -0.7, -0.2, 0.4]);
}

#[test]
fn test_inv_f32_3x3() {
    // 3x3 matrix: [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    // Using an invertible matrix
    let input = [1.0f32, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0];
    let mut output = [0.0f32; 9];

    let metadata = vec![1, 3, 2, 3, 3, 3, 1, 0];

    call_ops_inv(
        inv::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Verify A * A^{-1} = I by checking multiplication
    let a = input;
    let a_inv = output;
    let mut result = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[i * 3 + j] += a[i * 3 + k] * a_inv[k * 3 + j];
            }
        }
    }
    // Should be identity matrix
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert_eq!(approx(result.to_vec(), 4), identity.to_vec());
}

#[test]
fn test_inv_f32_identity() {
    // Identity matrix: inv(I) = I
    let input = [
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let mut output = [0.0f32; 16];

    let metadata = vec![1, 4, 2, 4, 4, 4, 1, 0];

    call_ops_inv(
        inv::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output.to_vec(), 4), input.to_vec());
}

#[test]
fn test_inv_f32_batch() {
    // Batch of 2 matrices:
    // Matrix 0: [[2, 0], [0, 2]], inv = [[0.5, 0], [0, 0.5]]
    // Matrix 1: [[1, 1], [0, 1]], inv = [[1, -1], [0, 1]]
    let input = [2.0f32, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 1.0];
    let mut output = [0.0f32; 8];

    // Metadata: [batch_size, n, ndim, shape..., strides..., offset]
    // shape = [2, 2, 2], strides = [4, 2, 1], offset = 0
    let metadata = vec![2, 2, 3, 2, 2, 2, 4, 2, 1, 0];

    call_ops_inv(
        inv::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    let expected = [0.5, 0.0, 0.0, 0.5, 1.0, -1.0, 0.0, 1.0];
    assert_eq!(approx(output.to_vec(), 4), expected.to_vec());
}
