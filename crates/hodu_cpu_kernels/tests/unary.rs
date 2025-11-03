use hodu_cpu_kernels::*;

fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t * b) / b).collect()
}

fn run_unary(input: &[f32], kernel: Kernel) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    let shape = vec![input.len()];
    let strides = vec![1];
    let num_els: usize = shape.iter().product();
    let num_dims = shape.len();
    let mut metadata = Vec::with_capacity(2 + num_dims * 2 + 1);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0); // offset
    call_unary(
        kernel,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();
    output
}

fn run_unary_to_bool(input: &[f32], kernel: Kernel) -> Vec<u8> {
    let mut output = vec![0u8; input.len()];
    let shape = vec![input.len()];
    let strides = vec![1];
    let num_els: usize = shape.iter().product();
    let num_dims = shape.len();
    let mut metadata = Vec::with_capacity(2 + num_dims * 2 + 1);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0); // offset
    call_unary(
        kernel,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();
    output
}

fn run_unary_scalar(input: &[f32], kernel: Kernel, scalar: f32) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    let shape = vec![input.len()];
    let strides = vec![1];
    let num_els: usize = shape.iter().product();
    let num_dims = shape.len();
    let mut metadata = Vec::with_capacity(2 + num_dims * 2 + 1);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0); // offset
    call_unary_scalar(
        kernel,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
        scalar,
    )
    .unwrap();
    output
}

fn run_unary_scalar_to_bool(input: &[f32], kernel: Kernel, scalar: f32) -> Vec<u8> {
    let mut output = vec![0u8; input.len()];
    let shape = vec![input.len()];
    let strides = vec![1];
    let num_els: usize = shape.iter().product();
    let num_dims = shape.len();
    let mut metadata = Vec::with_capacity(2 + num_dims * 2 + 1);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0); // offset
    call_unary_scalar(
        kernel,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
        scalar,
    )
    .unwrap();
    output
}

// unary - basic
#[test]
fn test_neg_f32() {
    let input = vec![1.0f32, -2.0, 3.0, -4.0];
    let output = run_unary(&input, neg::F32);
    assert_eq!(approx(output, 4), vec![-1.0, 2.0, -3.0, 4.0]);
}

#[test]
fn test_abs_f32() {
    let input = vec![-1.0f32, -2.0, 3.0, -4.0];
    let output = run_unary(&input, abs::F32);
    assert_eq!(approx(output, 4), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_sign_f32() {
    let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let output = run_unary(&input, sign::F32);
    assert_eq!(approx(output, 4), vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_square_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let output = run_unary(&input, square::F32);
    assert_eq!(approx(output, 4), vec![1.0, 4.0, 9.0, 16.0]);
}

#[test]
fn test_sqrt_f32() {
    let input = vec![1.0f32, 4.0, 9.0, 16.0];
    let output = run_unary(&input, sqrt::F32);
    assert_eq!(approx(output, 4), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_recip_f32() {
    let input = vec![1.0f32, 2.0, 4.0, 5.0];
    let output = run_unary(&input, recip::F32);
    assert_eq!(approx(output, 4), vec![1.0, 0.5, 0.25, 0.2]);
}

// unary - activation
#[test]
fn test_relu_f32() {
    let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let output = run_unary(&input, relu::F32);
    assert_eq!(approx(output, 4), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_sigmoid_f32() {
    let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let output = run_unary(&input, sigmoid::F32);
    // sigmoid(-2) ≈ 0.1192, sigmoid(-1) ≈ 0.2689, sigmoid(0) = 0.5, sigmoid(1) ≈ 0.7311, sigmoid(2) ≈ 0.8808
    assert_eq!(approx(output, 2), vec![0.12, 0.27, 0.5, 0.73, 0.88]);
}

#[test]
fn test_tanh_f32() {
    let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let output = run_unary(&input, tanh::F32);
    // tanh values
    let expected = input.iter().map(|x| x.tanh()).collect::<Vec<_>>();
    assert_eq!(approx(output, 2), approx(expected, 2));
}

#[test]
fn test_gelu_f32() {
    let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let output = run_unary(&input, gelu::F32);
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| 0.5 * x * (1.0 + (0.7978846 * (x + 0.044715 * x * x * x)).tanh()))
        .collect();
    assert_eq!(approx(output, 2), approx(expected, 2));
}

#[test]
fn test_softplus_f32() {
    let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let output = run_unary(&input, softplus::F32);
    // softplus(x) = ln(1 + exp(x))
    let expected: Vec<f32> = input.iter().map(|&x| (1.0 + x.exp()).ln()).collect();
    assert_eq!(approx(output, 2), approx(expected, 2));
}

#[test]
fn test_silu_f32() {
    let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let output = run_unary(&input, silu::F32);
    // silu(x) = x / (1 + exp(-x))
    let expected: Vec<f32> = input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
    assert_eq!(approx(output, 2), approx(expected, 2));
}

#[test]
fn test_mish_f32() {
    let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let output = run_unary(&input, mish::F32);
    // mish(x) = x * tanh(softplus(x))
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| {
            let softplus = (1.0 + x.exp()).ln();
            x * softplus.tanh()
        })
        .collect();
    assert_eq!(approx(output, 2), approx(expected, 2));
}

// unary - trigonometric
#[test]
fn test_sin_f32() {
    let input = vec![
        0.0f32,
        core::f32::consts::PI / 6.0,
        core::f32::consts::PI / 4.0,
        core::f32::consts::PI / 2.0,
    ];
    let output = run_unary(&input, sin::F32);
    let expected: Vec<f32> = input.iter().map(|x| x.sin()).collect();
    assert_eq!(approx(output, 2), approx(expected, 2));
}

#[test]
fn test_cos_f32() {
    let input = vec![
        0.0f32,
        core::f32::consts::PI / 6.0,
        core::f32::consts::PI / 4.0,
        core::f32::consts::PI / 2.0,
    ];
    let output = run_unary(&input, cos::F32);
    let expected: Vec<f32> = input.iter().map(|x| x.cos()).collect();
    assert_eq!(approx(output, 2), approx(expected, 2));
}

#[test]
fn test_tan_f32() {
    let input = vec![0.0f32, core::f32::consts::PI / 6.0, core::f32::consts::PI / 4.0];
    let output = run_unary(&input, tan::F32);
    let expected: Vec<f32> = input.iter().map(|x| x.tan()).collect();
    assert_eq!(approx(output, 2), approx(expected, 2));
}

// unary - exp
#[test]
fn test_exp_f32() {
    let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let output = run_unary(&input, exp::F32);
    let expected: Vec<f32> = input.iter().map(|x| x.exp()).collect();
    assert_eq!(approx(output, 2), approx(expected, 2));
}

#[test]
fn test_exp2_f32() {
    let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let output = run_unary(&input, exp2::F32);
    let expected: Vec<f32> = input.iter().map(|x| x.exp2()).collect();
    assert_eq!(approx(output, 2), approx(expected, 2));
}

#[test]
fn test_exp10_f32() {
    let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let output = run_unary(&input, exp10::F32);
    let expected: Vec<f32> = input.iter().map(|x| 10.0f32.powf(*x)).collect();
    assert_eq!(approx(output, 2), approx(expected, 2));
}

#[test]
fn test_ln_f32() {
    let input = vec![1.0f32, 2.0, core::f32::consts::E, 10.0];
    let output = run_unary(&input, ln::F32);
    let expected: Vec<f32> = input.iter().map(|x| x.ln()).collect();
    assert_eq!(approx(output, 2), approx(expected, 2));
}

#[test]
fn test_log2_f32() {
    let input = vec![1.0f32, 2.0, 4.0, 8.0];
    let output = run_unary(&input, log2::F32);
    let expected: Vec<f32> = input.iter().map(|x| x.log2()).collect();
    assert_eq!(approx(output, 2), approx(expected, 2));
}

#[test]
fn test_log10_f32() {
    let input = vec![1.0f32, 10.0, 100.0, 1000.0];
    let output = run_unary(&input, log10::F32);
    let expected: Vec<f32> = input.iter().map(|x| x.log10()).collect();
    assert_eq!(approx(output, 2), approx(expected, 2));
}

// unary logical
#[test]
fn test_logical_not_f32() {
    let input = vec![0.0f32, 1.0, 2.0, 0.0];
    let output = run_unary_to_bool(&input, logical_not::F32);
    assert_eq!(output, vec![1, 0, 0, 1]);
}

// unary with scalar - arithmetic
#[test]
fn test_add_scalar_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let output = run_unary_scalar(&input, add_scalar::F32, 5.0);
    assert_eq!(approx(output, 4), vec![6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn test_sub_scalar_f32() {
    let input = vec![5.0f32, 6.0, 7.0, 8.0];
    let output = run_unary_scalar(&input, sub_scalar::F32, 2.0);
    assert_eq!(approx(output, 4), vec![3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_mul_scalar_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let output = run_unary_scalar(&input, mul_scalar::F32, 3.0);
    assert_eq!(approx(output, 4), vec![3.0, 6.0, 9.0, 12.0]);
}

#[test]
fn test_div_scalar_f32() {
    let input = vec![10.0f32, 20.0, 30.0, 40.0];
    let output = run_unary_scalar(&input, div_scalar::F32, 2.0);
    assert_eq!(approx(output, 4), vec![5.0, 10.0, 15.0, 20.0]);
}

#[test]
fn test_pow_scalar_f32() {
    let input = vec![2.0f32, 3.0, 4.0, 5.0];
    let output = run_unary_scalar(&input, pow_scalar::F32, 2.0);
    assert_eq!(approx(output, 4), vec![4.0, 9.0, 16.0, 25.0]);
}

#[test]
fn test_maximum_scalar_f32() {
    let input = vec![1.0f32, 5.0, 3.0, 8.0];
    let output = run_unary_scalar(&input, maximum_scalar::F32, 4.0);
    assert_eq!(approx(output, 4), vec![4.0, 5.0, 4.0, 8.0]);
}

#[test]
fn test_minimum_scalar_f32() {
    let input = vec![1.0f32, 5.0, 3.0, 8.0];
    let output = run_unary_scalar(&input, minimum_scalar::F32, 4.0);
    assert_eq!(approx(output, 4), vec![1.0, 4.0, 3.0, 4.0]);
}

// cmp with scalar
#[test]
fn test_eq_scalar_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let output = run_unary_scalar_to_bool(&input, eq_scalar::F32, 3.0);
    assert_eq!(output, vec![0, 0, 1, 0]);
}

#[test]
fn test_ne_scalar_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let output = run_unary_scalar_to_bool(&input, ne_scalar::F32, 3.0);
    assert_eq!(output, vec![1, 1, 0, 1]);
}

#[test]
fn test_lt_scalar_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let output = run_unary_scalar_to_bool(&input, lt_scalar::F32, 3.0);
    assert_eq!(output, vec![1, 1, 0, 0]);
}

#[test]
fn test_le_scalar_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let output = run_unary_scalar_to_bool(&input, le_scalar::F32, 3.0);
    assert_eq!(output, vec![1, 1, 1, 0]);
}

#[test]
fn test_gt_scalar_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let output = run_unary_scalar_to_bool(&input, gt_scalar::F32, 3.0);
    assert_eq!(output, vec![0, 0, 0, 1]);
}

#[test]
fn test_ge_scalar_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let output = run_unary_scalar_to_bool(&input, ge_scalar::F32, 3.0);
    assert_eq!(output, vec![0, 0, 1, 1]);
}
