use hodu_cpu_kernels::*;

fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t * b) / b).collect()
}

fn run_binary(lhs: &[f32], rhs: &[f32], kernel: Kernel) -> Vec<f32> {
    assert_eq!(lhs.len(), rhs.len());
    let mut output = vec![0.0f32; lhs.len()];
    let shape = vec![lhs.len()];
    let strides = vec![1];
    call_binary(
        kernel,
        lhs.as_ptr() as *const std::ffi::c_void,
        rhs.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &shape,
        &strides,
        &strides,
        0,
        0,
    );
    output
}

fn run_binary_to_bool(lhs: &[f32], rhs: &[f32], kernel: Kernel) -> Vec<u8> {
    assert_eq!(lhs.len(), rhs.len());
    let mut output = vec![0u8; lhs.len()];
    let shape = vec![lhs.len()];
    let strides = vec![1];
    call_binary(
        kernel,
        lhs.as_ptr() as *const std::ffi::c_void,
        rhs.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &shape,
        &strides,
        &strides,
        0,
        0,
    );
    output
}

// binary
#[test]
fn test_add_f32() {
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![5.0f32, 6.0, 7.0, 8.0];
    let output = run_binary(&lhs, &rhs, add::F32);
    assert_eq!(approx(output, 4), vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_sub_f32() {
    let lhs = vec![10.0f32, 20.0, 30.0, 40.0];
    let rhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let output = run_binary(&lhs, &rhs, sub::F32);
    assert_eq!(approx(output, 4), vec![9.0, 18.0, 27.0, 36.0]);
}

#[test]
fn test_mul_f32() {
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![2.0f32, 3.0, 4.0, 5.0];
    let output = run_binary(&lhs, &rhs, mul::F32);
    assert_eq!(approx(output, 4), vec![2.0, 6.0, 12.0, 20.0]);
}

#[test]
fn test_div_f32() {
    let lhs = vec![10.0f32, 20.0, 30.0, 40.0];
    let rhs = vec![2.0f32, 4.0, 5.0, 8.0];
    let output = run_binary(&lhs, &rhs, div::F32);
    assert_eq!(approx(output, 4), vec![5.0, 5.0, 6.0, 5.0]);
}

#[test]
fn test_pow_f32() {
    let lhs = vec![2.0f32, 3.0, 4.0, 5.0];
    let rhs = vec![2.0f32, 2.0, 2.0, 2.0];
    let output = run_binary(&lhs, &rhs, pow::F32);
    assert_eq!(approx(output, 4), vec![4.0, 9.0, 16.0, 25.0]);
}

#[test]
fn test_maximum_f32() {
    let lhs = vec![1.0f32, 5.0, 3.0, 8.0];
    let rhs = vec![3.0f32, 2.0, 7.0, 6.0];
    let output = run_binary(&lhs, &rhs, maximum::F32);
    assert_eq!(approx(output, 4), vec![3.0, 5.0, 7.0, 8.0]);
}

#[test]
fn test_minimum_f32() {
    let lhs = vec![1.0f32, 5.0, 3.0, 8.0];
    let rhs = vec![3.0f32, 2.0, 7.0, 6.0];
    let output = run_binary(&lhs, &rhs, minimum::F32);
    assert_eq!(approx(output, 4), vec![1.0, 2.0, 3.0, 6.0]);
}

// binary logical
#[test]
fn test_logical_and_f32() {
    let lhs = vec![0.0f32, 0.0, 1.0, 2.0];
    let rhs = vec![0.0f32, 3.0, 0.0, 4.0];
    let output = run_binary_to_bool(&lhs, &rhs, logical_and::F32);
    assert_eq!(output, vec![0, 0, 0, 1]);
}

#[test]
fn test_logical_or_f32() {
    let lhs = vec![0.0f32, 0.0, 1.0, 2.0];
    let rhs = vec![0.0f32, 3.0, 0.0, 4.0];
    let output = run_binary_to_bool(&lhs, &rhs, logical_or::F32);
    assert_eq!(output, vec![0, 1, 1, 1]);
}

#[test]
fn test_logical_xor_f32() {
    let lhs = vec![0.0f32, 0.0, 1.0, 2.0];
    let rhs = vec![0.0f32, 3.0, 0.0, 4.0];
    let output = run_binary_to_bool(&lhs, &rhs, logical_xor::F32);
    assert_eq!(output, vec![0, 1, 1, 0]);
}

// cmp
#[test]
fn test_eq_f32() {
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![1.0f32, 3.0, 3.0, 5.0];
    let output = run_binary_to_bool(&lhs, &rhs, eq::F32);
    assert_eq!(output, vec![1, 0, 1, 0]);
}

#[test]
fn test_ne_f32() {
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![1.0f32, 3.0, 3.0, 5.0];
    let output = run_binary_to_bool(&lhs, &rhs, ne::F32);
    assert_eq!(output, vec![0, 1, 0, 1]);
}

#[test]
fn test_lt_f32() {
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![2.0f32, 2.0, 2.0, 3.0];
    let output = run_binary_to_bool(&lhs, &rhs, lt::F32);
    assert_eq!(output, vec![1, 0, 0, 0]);
}

#[test]
fn test_le_f32() {
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![2.0f32, 2.0, 2.0, 3.0];
    let output = run_binary_to_bool(&lhs, &rhs, le::F32);
    assert_eq!(output, vec![1, 1, 0, 0]);
}

#[test]
fn test_gt_f32() {
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![2.0f32, 2.0, 2.0, 3.0];
    let output = run_binary_to_bool(&lhs, &rhs, gt::F32);
    assert_eq!(output, vec![0, 0, 1, 1]);
}

#[test]
fn test_ge_f32() {
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![2.0f32, 2.0, 2.0, 3.0];
    let output = run_binary_to_bool(&lhs, &rhs, ge::F32);
    assert_eq!(output, vec![0, 1, 1, 1]);
}

// Edge cases
#[test]
fn test_div_by_zero_f32() {
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs = vec![1.0f32, 0.0, 3.0, 0.0];
    let output = run_binary(&lhs, &rhs, div::F32);
    // Division by zero should produce infinity
    assert_eq!(output[0], 1.0);
    assert!(output[1].is_infinite());
    assert_eq!(output[2], 1.0);
    assert!(output[3].is_infinite());
}

#[test]
fn test_negative_values_f32() {
    let lhs = vec![-1.0f32, -2.0, -3.0, -4.0];
    let rhs = vec![1.0f32, 2.0, 3.0, 4.0];
    let output = run_binary(&lhs, &rhs, add::F32);
    assert_eq!(approx(output, 4), vec![0.0, 0.0, 0.0, 0.0]);
}
