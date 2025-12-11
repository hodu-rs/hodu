use hodu_cuda_kernels::{kernel::Kernels, kernels::*};

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

fn run_unary<T: cudarc::driver::DeviceRepr + Clone>(kernel: Kernel, input: &[T]) -> Vec<T> {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let mut output = unsafe { stream.alloc::<T>(input.len()).unwrap() };

    let shape = vec![input.len()];
    let strides = vec![1];
    let num_els = input.len();
    let num_dims = 1;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0); // offset

    call_ops_unary::<T, T>(kernel, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut result = vec![unsafe { core::mem::zeroed() }; input.len()];
    stream.memcpy_dtoh(&output, &mut result).unwrap();
    result
}

fn run_unary_scalar<T: cudarc::driver::DeviceRepr + Clone>(kernel: Kernel, input: &[T], scalar: T) -> Vec<T> {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let mut output = unsafe { stream.alloc::<T>(input.len()).unwrap() };

    let shape = vec![input.len()];
    let strides = vec![1];
    let num_els = input.len();
    let num_dims = 1;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0); // offset

    call_ops_unary_scalar::<T, T>(kernel, &kernels, &device, &input_dev, &mut output, &metadata, scalar).unwrap();

    let mut result = vec![unsafe { core::mem::zeroed() }; input.len()];
    stream.memcpy_dtoh(&output, &mut result).unwrap();
    result
}

fn run_unary_logical<T: cudarc::driver::DeviceRepr + Clone>(kernel: Kernel, input: &[T], scalar: T) -> Vec<bool> {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let mut output = unsafe { stream.alloc::<bool>(input.len()).unwrap() };

    let shape = vec![input.len()];
    let strides = vec![1];
    let num_els = input.len();
    let num_dims = 1;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0); // offset

    call_ops_unary_scalar::<T, bool>(kernel, &kernels, &device, &input_dev, &mut output, &metadata, scalar).unwrap();

    let mut result = vec![false; input.len()];
    stream.memcpy_dtoh(&output, &mut result).unwrap();
    result
}

// Scalar comparison operations
#[test]
fn unary_scalar_cmp_f32() {
    let input: Vec<f32> = vec![1.0f32, 2.0, 3.0];
    let scalar = 2.0f32;

    let eq_results = run_unary_logical(eq_scalar::F32, &input, scalar);
    let ne_results = run_unary_logical(ne_scalar::F32, &input, scalar);
    let lt_results = run_unary_logical(lt_scalar::F32, &input, scalar);
    let le_results = run_unary_logical(le_scalar::F32, &input, scalar);
    let gt_results = run_unary_logical(gt_scalar::F32, &input, scalar);
    let ge_results = run_unary_logical(ge_scalar::F32, &input, scalar);

    assert_eq!(eq_results, vec![false, true, false]);
    assert_eq!(ne_results, vec![true, false, true]);
    assert_eq!(lt_results, vec![true, false, false]);
    assert_eq!(le_results, vec![true, true, false]);
    assert_eq!(gt_results, vec![false, false, true]);
    assert_eq!(ge_results, vec![false, true, true]);
}

// Basic math operations
#[test]
fn unary_basic_math_f32() {
    let input: Vec<f32> = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];

    let neg_results = run_unary(neg::F32, &input);
    let expected_neg: Vec<f32> = input.iter().map(|x| -x).collect();
    assert_eq!(approx(neg_results, 6), approx(expected_neg, 6));

    let abs_results = run_unary(abs::F32, &input);
    let expected_abs: Vec<f32> = input.iter().map(|x| x.abs()).collect();
    assert_eq!(approx(abs_results, 6), approx(expected_abs, 6));

    let sign_results = run_unary(sign::F32, &input);
    let expected_sign: Vec<f32> = input
        .iter()
        .map(|x| {
            if *x > 0.0 {
                1.0
            } else if *x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect();
    assert_eq!(approx(sign_results, 6), approx(expected_sign, 6));
}

#[test]
fn unary_square_sqrt_recip_f32() {
    let input: Vec<f32> = vec![1.0f32, 2.0, 4.0, 9.0];

    let square_results = run_unary(square::F32, &input);
    let expected_square: Vec<f32> = input.iter().map(|x| x * x).collect();
    assert_eq!(approx(square_results, 6), approx(expected_square, 6));

    let sqrt_results = run_unary(sqrt::F32, &input);
    let expected_sqrt: Vec<f32> = input.iter().map(|x| x.sqrt()).collect();
    assert_eq!(approx(sqrt_results, 6), approx(expected_sqrt, 6));

    let recip_results = run_unary(recip::F32, &input);
    let expected_recip: Vec<f32> = input.iter().map(|x| 1.0 / x).collect();
    assert_eq!(approx(recip_results, 6), approx(expected_recip, 6));
}

// Activation functions
#[test]
fn unary_relu_f32() {
    let input: Vec<f32> = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];

    let results = run_unary(relu::F32, &input);
    let expected: Vec<f32> = input.iter().map(|x| x.max(0.0)).collect();
    assert_eq!(approx(results, 6), approx(expected, 6));
}

#[test]
fn unary_sigmoid_f32() {
    let input: Vec<f32> = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];

    let results = run_unary(sigmoid::F32, &input);
    let expected: Vec<f32> = input.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    assert_eq!(approx(results, 6), approx(expected, 6));
}

#[test]
fn unary_tanh_f32() {
    let input: Vec<f32> = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];

    let results = run_unary(tanh::F32, &input);
    let expected: Vec<f32> = input.iter().map(|x| x.tanh()).collect();
    assert_eq!(approx(results, 6), approx(expected, 6));
}

#[test]
fn unary_gelu_f32() {
    let input: Vec<f32> = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];

    let results = run_unary(gelu::F32, &input);
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let expected: Vec<f32> = input
        .iter()
        .map(|x| {
            let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
            0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
        })
        .collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn unary_softplus_f32() {
    let input: Vec<f32> = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];

    let results = run_unary(softplus::F32, &input);
    let expected: Vec<f32> = input.iter().map(|x| (1.0 + x.exp()).ln()).collect();
    assert_eq!(approx(results, 6), approx(expected, 6));
}

#[test]
fn unary_silu_f32() {
    let input: Vec<f32> = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];

    let results = run_unary(silu::F32, &input);
    let expected: Vec<f32> = input.iter().map(|x| x / (1.0 + (-x).exp())).collect();
    assert_eq!(approx(results, 6), approx(expected, 6));
}

#[test]
fn unary_mish_f32() {
    let input: Vec<f32> = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];

    let results = run_unary(mish::F32, &input);
    let expected: Vec<f32> = input.iter().map(|x| x * ((1.0 + x.exp()).ln()).tanh()).collect();
    assert_eq!(approx(results, 6), approx(expected, 6));
}

// Trigonometric functions
#[test]
fn unary_trig_f32() {
    let input: Vec<f32> = vec![
        0.0f32,
        std::f32::consts::PI / 6.0,
        std::f32::consts::PI / 4.0,
        std::f32::consts::PI / 3.0,
    ];

    let sin_results = run_unary(sin::F32, &input);
    let expected_sin: Vec<f32> = input.iter().map(|x| x.sin()).collect();
    assert_eq!(approx(sin_results, 6), approx(expected_sin, 6));

    let cos_results = run_unary(cos::F32, &input);
    let expected_cos: Vec<f32> = input.iter().map(|x| x.cos()).collect();
    assert_eq!(approx(cos_results, 6), approx(expected_cos, 6));

    let tan_results = run_unary(tan::F32, &input);
    let expected_tan: Vec<f32> = input.iter().map(|x| x.tan()).collect();
    assert_eq!(approx(tan_results, 6), approx(expected_tan, 6));
}

#[test]
fn unary_inverse_trig_f32() {
    let input: Vec<f32> = vec![-0.5f32, 0.0, 0.5, 1.0];

    let asin_results = run_unary(asin::F32, &input);
    let expected_asin: Vec<f32> = input.iter().map(|x| x.asin()).collect();
    assert_eq!(approx(asin_results, 4), approx(expected_asin, 4));

    let acos_results = run_unary(acos::F32, &input);
    let expected_acos: Vec<f32> = input.iter().map(|x| x.acos()).collect();
    assert_eq!(approx(acos_results, 4), approx(expected_acos, 4));

    let atan_input: Vec<f32> = vec![-1.0f32, 0.0, 1.0, 2.0];
    let atan_results = run_unary(atan::F32, &atan_input);
    let expected_atan: Vec<f32> = atan_input.iter().map(|x| x.atan()).collect();
    assert_eq!(approx(atan_results, 4), approx(expected_atan, 4));
}

// Hyperbolic functions
#[test]
fn unary_hyperbolic_f32() {
    let input: Vec<f32> = vec![-1.0f32, 0.0, 1.0, 2.0];

    let sinh_results = run_unary(sinh::F32, &input);
    let expected_sinh: Vec<f32> = input.iter().map(|x| x.sinh()).collect();
    assert_eq!(approx(sinh_results, 4), approx(expected_sinh, 4));

    let cosh_results = run_unary(cosh::F32, &input);
    let expected_cosh: Vec<f32> = input.iter().map(|x| x.cosh()).collect();
    assert_eq!(approx(cosh_results, 4), approx(expected_cosh, 4));
}

#[test]
fn unary_inverse_hyperbolic_f32() {
    let asinh_input: Vec<f32> = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let asinh_results = run_unary(asinh::F32, &asinh_input);
    let expected_asinh: Vec<f32> = asinh_input.iter().map(|x| x.asinh()).collect();
    assert_eq!(approx(asinh_results, 4), approx(expected_asinh, 4));

    let acosh_input: Vec<f32> = vec![1.0f32, 2.0, 3.0, 4.0];
    let acosh_results = run_unary(acosh::F32, &acosh_input);
    let expected_acosh: Vec<f32> = acosh_input.iter().map(|x| x.acosh()).collect();
    assert_eq!(approx(acosh_results, 4), approx(expected_acosh, 4));

    let atanh_input: Vec<f32> = vec![-0.5f32, 0.0, 0.5];
    let atanh_results = run_unary(atanh::F32, &atanh_input);
    let expected_atanh: Vec<f32> = atanh_input.iter().map(|x| x.atanh()).collect();
    assert_eq!(approx(atanh_results, 4), approx(expected_atanh, 4));
}

// Exponential and logarithmic functions
#[test]
fn unary_exp_f32() {
    let input: Vec<f32> = vec![0.0f32, 1.0, 2.0, 3.0];

    let exp_results = run_unary(exp::F32, &input);
    let expected_exp: Vec<f32> = input.iter().map(|x| x.exp()).collect();
    assert_eq!(approx(exp_results, 3), approx(expected_exp, 3));

    let exp2_results = run_unary(exp2::F32, &input);
    let expected_exp2: Vec<f32> = input.iter().map(|x| x.exp2()).collect();
    assert_eq!(approx(exp2_results, 3), approx(expected_exp2, 3));

    let exp10_results = run_unary(exp10::F32, &input);
    let expected_exp10: Vec<f32> = input.iter().map(|x| 10.0f32.powf(*x)).collect();
    assert_eq!(approx(exp10_results, 3), approx(expected_exp10, 3));
}

#[test]
fn unary_log_f32() {
    let input: Vec<f32> = vec![1.0f32, 2.0, 10.0, 100.0];

    let ln_results = run_unary(ln::F32, &input);
    let expected_ln: Vec<f32> = input.iter().map(|x| x.ln()).collect();
    assert_eq!(approx(ln_results, 6), approx(expected_ln, 6));

    let log2_results = run_unary(log2::F32, &input);
    let expected_log2: Vec<f32> = input.iter().map(|x| x.log2()).collect();
    assert_eq!(approx(log2_results, 6), approx(expected_log2, 6));

    let log10_results = run_unary(log10::F32, &input);
    let expected_log10: Vec<f32> = input.iter().map(|x| x.log10()).collect();
    assert_eq!(approx(log10_results, 6), approx(expected_log10, 6));
}

#[test]
fn unary_ceil_f32() {
    let input: Vec<f32> = vec![1.1f32, 2.5, -1.1, -2.9, 3.0];
    let results = run_unary(ceil::F32, &input);
    assert_eq!(results, vec![2.0, 3.0, -1.0, -2.0, 3.0]);
}

#[test]
fn unary_floor_f32() {
    let input: Vec<f32> = vec![1.1f32, 2.5, -1.1, -2.9, 3.0];
    let results = run_unary(floor::F32, &input);
    assert_eq!(results, vec![1.0, 2.0, -2.0, -3.0, 3.0]);
}

#[test]
fn unary_round_f32() {
    let input: Vec<f32> = vec![1.4f32, 1.5, 2.5, -1.4, -1.5, -2.5];
    let results = run_unary(round::F32, &input);
    assert_eq!(results, vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
}

#[test]
fn unary_erf_f32() {
    let input: Vec<f32> = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run_unary(erf::F32, &input);
    // erf values calculated from standard math library
    let expected = vec![-0.9953, -0.8427, 0.0, 0.8427, 0.9953];
    assert_eq!(approx(results, 4), expected);
}

// Logical operations
#[test]
fn unary_logical_not() {
    let input: Vec<bool> = vec![true, false, true, false];

    let results = run_unary(logical_not::BOOL, &input);
    let expected: Vec<bool> = input.iter().map(|x| !x).collect();
    assert_eq!(results, expected);
}

// Scalar arithmetic operations
#[test]
fn unary_scalar_arithmetic_f32() {
    let input: Vec<f32> = vec![1.0f32, 2.0, 3.0];
    let scalar = 2.0f32;

    let add_results = run_unary_scalar(add_scalar::F32, &input, scalar);
    let expected_add: Vec<f32> = input.iter().map(|x| x + scalar).collect();
    assert_eq!(approx(add_results, 6), approx(expected_add, 6));

    let sub_results = run_unary_scalar(sub_scalar::F32, &input, scalar);
    let expected_sub: Vec<f32> = input.iter().map(|x| x - scalar).collect();
    assert_eq!(approx(sub_results, 6), approx(expected_sub, 6));

    let mul_results = run_unary_scalar(mul_scalar::F32, &input, scalar);
    let expected_mul: Vec<f32> = input.iter().map(|x| x * scalar).collect();
    assert_eq!(approx(mul_results, 6), approx(expected_mul, 6));

    let div_results = run_unary_scalar(div_scalar::F32, &input, scalar);
    let expected_div: Vec<f32> = input.iter().map(|x| x / scalar).collect();
    assert_eq!(approx(div_results, 6), approx(expected_div, 6));

    let pow_results = run_unary_scalar(pow_scalar::F32, &input, scalar);
    let expected_pow: Vec<f32> = input.iter().map(|x| x.powf(scalar)).collect();
    assert_eq!(approx(pow_results, 6), approx(expected_pow, 6));
}

#[test]
fn unary_scalar_minmax_f32() {
    let input: Vec<f32> = vec![1.0f32, 2.0, 3.0];
    let scalar = 2.0f32;

    let maximum_results = run_unary_scalar(maximum_scalar::F32, &input, scalar);
    let expected_maximum: Vec<f32> = input.iter().map(|x| x.max(scalar)).collect();
    assert_eq!(approx(maximum_results, 6), approx(expected_maximum, 6));

    let minimum_results = run_unary_scalar(minimum_scalar::F32, &input, scalar);
    let expected_minimum: Vec<f32> = input.iter().map(|x| x.min(scalar)).collect();
    assert_eq!(approx(minimum_results, 6), approx(expected_minimum, 6));
}

// Parametric activations
#[test]
fn unary_leaky_relu_f32() {
    let input: Vec<f32> = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let alpha = 0.01f32;

    let results = run_unary_scalar(leaky_relu::F32, &input, alpha);
    let expected: Vec<f32> = input.iter().map(|x| if *x > 0.0 { *x } else { alpha * x }).collect();
    assert_eq!(approx(results, 6), approx(expected, 6));
}

#[test]
fn unary_elu_f32() {
    let input: Vec<f32> = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let alpha = 1.0f32;

    let results = run_unary_scalar(elu::F32, &input, alpha);
    let expected: Vec<f32> = input
        .iter()
        .map(|x| if *x > 0.0 { *x } else { alpha * (x.exp() - 1.0) })
        .collect();
    assert_eq!(approx(results, 6), approx(expected, 6));
}

#[test]
fn unary_prelu_f32() {
    let input: Vec<f32> = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let alpha = 0.25f32;

    let results = run_unary_scalar(prelu::F32, &input, alpha);
    let expected: Vec<f32> = input.iter().map(|x| if *x > 0.0 { *x } else { alpha * x }).collect();
    assert_eq!(approx(results, 6), approx(expected, 6));
}

#[test]
fn unary_hardsigmoid_f32() {
    let input: Vec<f32> = vec![-4.0f32, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0];

    let results = run_unary(hardsigmoid::F32, &input);
    let expected: Vec<f32> = input.iter().map(|x| ((x + 3.0) / 6.0).clamp(0.0, 1.0)).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn unary_hardsilu_f32() {
    let input: Vec<f32> = vec![-4.0f32, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0];

    let results = run_unary(hardsilu::F32, &input);
    let expected: Vec<f32> = input.iter().map(|x| x * ((x + 3.0) / 6.0).clamp(0.0, 1.0)).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}
