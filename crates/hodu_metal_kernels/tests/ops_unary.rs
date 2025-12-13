use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_ops_unary, call_ops_unary_scalar, Kernel, *},
    metal::{create_command_buffer, Buffer, Device},
    utils::{BufferOffset, EncoderParam},
    RESOURCE_OPTIONS,
};
use std::ffi::c_void;

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

fn new_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let options = RESOURCE_OPTIONS;
    let ptr = data.as_ptr() as *const c_void;
    let size = std::mem::size_of_val(data);
    device.new_buffer_with_data(ptr, size, options).unwrap()
}

fn device() -> Device {
    Device::system_default().unwrap()
}

fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t * b) / b).collect()
}

fn run<T: Clone>(v: &[T], kernel: Kernel) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);

    let shape = vec![v.len()];
    let strides = vec![1];
    let num_els = v.len();
    let num_dims = shape.len();

    let mut metadata = Vec::with_capacity(2 + num_dims * 2 + 1);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0); // offset

    call_ops_unary(
        kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input),
        &output,
        &metadata,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, v.len())
}

fn run_scalar<T: Clone + EncoderParam>(v: &[T], kernel: Kernel, scalar: T) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);

    let shape = vec![v.len()];
    let strides = vec![1];
    let num_els = v.len();
    let num_dims = shape.len();

    let mut metadata = Vec::with_capacity(2 + num_dims * 2 + 1);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0); // offset

    call_ops_unary_scalar(
        kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input),
        &output,
        &metadata,
        scalar,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, v.len())
}

fn run_scalar_logical<T: Clone + EncoderParam>(v: &[T], kernel: Kernel, scalar: T) -> Vec<bool> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;
    let input = new_buffer(&device, v);
    let output = device
        .new_buffer(v.len() * std::mem::size_of::<bool>(), options)
        .unwrap();

    let shape = vec![v.len()];
    let strides = vec![1];
    let num_els = v.len();
    let num_dims = shape.len();

    let mut metadata = Vec::with_capacity(2 + num_dims * 2 + 1);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0); // offset

    call_ops_unary_scalar(
        kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input),
        &output,
        &metadata,
        scalar,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, v.len())
}

// Basic math operations
#[test]
fn abs_f32() {
    let v = vec![-1.0f32, -2.0, 3.0, -4.0];
    let results = run(&v, abs::F32);
    assert_eq!(approx(results, 4), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn neg_f32() {
    let v = vec![1.0f32, -2.0, 3.0, -4.0];
    let results = run(&v, neg::F32);
    assert_eq!(approx(results, 4), vec![-1.0, 2.0, -3.0, 4.0]);
}

#[test]
fn sign_f32() {
    let v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run(&v, sign::F32);
    assert_eq!(approx(results, 4), vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn square_f32() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0];
    let results = run(&v, square::F32);
    assert_eq!(approx(results, 4), vec![1.0, 4.0, 9.0, 16.0]);
}

#[test]
fn sqrt_f32() {
    let v = vec![1.0f32, 4.0, 9.0, 16.0];
    let results = run(&v, sqrt::F32);
    assert_eq!(approx(results, 4), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn recip_f32() {
    let v = vec![1.0f32, 2.0, 4.0, 5.0];
    let results = run(&v, recip::F32);
    assert_eq!(approx(results, 4), vec![1.0, 0.5, 0.25, 0.2]);
}

// Activation functions
#[test]
fn relu_f32() {
    let v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run(&v, relu::F32);
    assert_eq!(approx(results, 4), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn sigmoid_f32() {
    let v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run(&v, sigmoid::F32);
    let expected: Vec<_> = v.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn hardsigmoid_f32() {
    let v = vec![-4.0f32, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0];
    let results = run(&v, hardsigmoid::F32);
    let expected: Vec<_> = v.iter().map(|x| ((x + 3.0) / 6.0).clamp(0.0, 1.0)).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn tanh_f32() {
    let v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run(&v, tanh::F32);
    let expected: Vec<_> = v.iter().map(|v| v.tanh()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn gelu_f32() {
    let v: Vec<f32> = vec![-3.0f32, -1.0, 0., 1., 2., 3.0];
    let results = run(&v, gelu::F32);
    let expected: Vec<f32> = vec![-0.004, -0.159, 0.0, 0.841, 1.955, 2.996];
    assert_eq!(approx(results, 3), expected);
}

#[test]
fn softplus_f32() {
    let v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run(&v, softplus::F32);
    let expected: Vec<_> = v.iter().map(|v| (1.0 + v.exp()).ln()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn silu_f32() {
    let v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run(&v, silu::F32);
    let expected: Vec<_> = v.iter().map(|v| v / (1.0 + (-v).exp())).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn hardsilu_f32() {
    let v = vec![-4.0f32, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0];
    let results = run(&v, hardsilu::F32);
    let expected: Vec<_> = v.iter().map(|x| x * ((x + 3.0) / 6.0).clamp(0.0, 1.0)).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn mish_f32() {
    let v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run(&v, mish::F32);
    let expected: Vec<_> = v.iter().map(|v| v * (1.0 + v.exp()).ln().tanh()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn softsign_f32() {
    let v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run(&v, softsign::F32);
    // softsign(x) = x / (1 + |x|)
    let expected: Vec<_> = v.iter().map(|x| x / (1.0 + x.abs())).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn selu_f32() {
    let v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run(&v, selu::F32);
    // selu(x) = scale * (max(0,x) + min(0, alpha*(exp(x)-1)))
    let alpha = 1.6732632423543772848170429916717f32;
    let scale = 1.0507009873554804934193349852946f32;
    let expected: Vec<_> = v
        .iter()
        .map(|&x| {
            if x > 0.0 {
                scale * x
            } else {
                scale * alpha * (x.exp() - 1.0)
            }
        })
        .collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn celu_f32() {
    let v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run(&v, celu::F32);
    // celu(x) = max(0,x) + min(0, exp(x)-1)
    let expected: Vec<_> = v.iter().map(|&x| x.max(0.0) + (x.exp() - 1.0).min(0.0)).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

// Trigonometric functions
#[test]
fn sin_f32() {
    let v = vec![0.0f32, 1.0, 2.0, 3.0];
    let results = run(&v, sin::F32);
    let expected: Vec<_> = v.iter().map(|v| v.sin()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn cos_f32() {
    let v = vec![1.0f32, 2.0, 3.0];
    let results = run(&v, cos::F32);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403, -0.4161, -0.99]);
    assert_eq!(approx(expected, 4), vec![0.5403, -0.4161, -0.99]);
}

#[test]
fn tan_f32() {
    let v = vec![0.0f32, 0.5, 1.0];
    let results = run(&v, tan::F32);
    let expected: Vec<_> = v.iter().map(|v| v.tan()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn asin_f32() {
    let v = vec![-0.5f32, 0.0, 0.5, 1.0];
    let results = run(&v, asin::F32);
    let expected: Vec<_> = v.iter().map(|v| v.asin()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn acos_f32() {
    let v = vec![-0.5f32, 0.0, 0.5, 1.0];
    let results = run(&v, acos::F32);
    let expected: Vec<_> = v.iter().map(|v| v.acos()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn atan_f32() {
    let v = vec![-1.0f32, 0.0, 1.0, 2.0];
    let results = run(&v, atan::F32);
    let expected: Vec<_> = v.iter().map(|v| v.atan()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

// Hyperbolic functions
#[test]
fn sinh_f32() {
    let v = vec![-1.0f32, 0.0, 1.0, 2.0];
    let results = run(&v, sinh::F32);
    let expected: Vec<_> = v.iter().map(|v| v.sinh()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn cosh_f32() {
    let v = vec![-1.0f32, 0.0, 1.0, 2.0];
    let results = run(&v, cosh::F32);
    let expected: Vec<_> = v.iter().map(|v| v.cosh()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn asinh_f32() {
    let v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run(&v, asinh::F32);
    let expected: Vec<_> = v.iter().map(|v| v.asinh()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn acosh_f32() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0];
    let results = run(&v, acosh::F32);
    let expected: Vec<_> = v.iter().map(|v| v.acosh()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn atanh_f32() {
    let v = vec![-0.5f32, 0.0, 0.5];
    let results = run(&v, atanh::F32);
    let expected: Vec<_> = v.iter().map(|v| v.atanh()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

// Exponential and logarithmic functions
#[test]
fn exp_f32() {
    let v = vec![0.0f32, 1.0, 2.0];
    let results = run(&v, exp::F32);
    let expected: Vec<_> = v.iter().map(|v| v.exp()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn exp2_f32() {
    let v = vec![0.0f32, 1.0, 2.0, 3.0];
    let results = run(&v, exp2::F32);
    let expected: Vec<_> = v.iter().map(|v| v.exp2()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn exp10_f32() {
    let v = vec![0.0f32, 1.0, 2.0];
    let results = run(&v, exp10::F32);
    assert_eq!(approx(results, 4), vec![1.0, 10.0, 100.0]);
}

#[test]
fn ln_f32() {
    let v = vec![1.0f32, 2.0, 10.0];
    let results = run(&v, ln::F32);
    let expected: Vec<_> = v.iter().map(|v| v.ln()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn log2_f32() {
    let v = vec![1.0f32, 2.0, 4.0, 8.0];
    let results = run(&v, log2::F32);
    let expected: Vec<_> = v.iter().map(|v| v.log2()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn log10_f32() {
    let v = vec![1.0f32, 10.0, 100.0];
    let results = run(&v, log10::F32);
    let expected: Vec<_> = v.iter().map(|v| v.log10()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn ceil_f32() {
    let v = vec![1.1f32, 2.5, -1.1, -2.9, 3.0];
    let results = run(&v, ceil::F32);
    assert_eq!(results, vec![2.0, 3.0, -1.0, -2.0, 3.0]);
}

#[test]
fn floor_f32() {
    let v = vec![1.1f32, 2.5, -1.1, -2.9, 3.0];
    let results = run(&v, floor::F32);
    assert_eq!(results, vec![1.0, 2.0, -2.0, -3.0, 3.0]);
}

#[test]
fn round_f32() {
    let v = vec![1.4f32, 1.5, 2.5, -1.4, -1.5, -2.5];
    let results = run(&v, round::F32);
    assert_eq!(results, vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
}

#[test]
fn erf_f32() {
    let v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run(&v, erf::F32);
    // erf values calculated from standard math library
    let expected = vec![-0.9953, -0.8427, 0.0, 0.8427, 0.9953];
    assert_eq!(approx(results, 4), expected);
}

// Scalar comparison operations
#[test]
fn unary_scalar_cmp_f32() {
    let input: Vec<f32> = vec![1.0f32, 2.0, 3.0];
    let scalar = 2.0f32;

    let eq_results = run_scalar_logical(&input, eq_scalar::F32, scalar);
    let ne_results = run_scalar_logical(&input, ne_scalar::F32, scalar);
    let lt_results = run_scalar_logical(&input, lt_scalar::F32, scalar);
    let le_results = run_scalar_logical(&input, le_scalar::F32, scalar);
    let gt_results = run_scalar_logical(&input, gt_scalar::F32, scalar);
    let ge_results = run_scalar_logical(&input, ge_scalar::F32, scalar);

    assert_eq!(eq_results, vec![false, true, false]);
    assert_eq!(ne_results, vec![true, false, true]);
    assert_eq!(lt_results, vec![true, false, false]);
    assert_eq!(le_results, vec![true, true, false]);
    assert_eq!(gt_results, vec![false, false, true]);
    assert_eq!(ge_results, vec![false, true, true]);
}

// Scalar arithmetic operations
#[test]
fn unary_scalar_arithmetic_f32() {
    let input: Vec<f32> = vec![1.0f32, 2.0, 3.0];
    let scalar = 2.0f32;

    let add_results = run_scalar(&input, add_scalar::F32, scalar);
    let expected_add: Vec<f32> = input.iter().map(|x| x + scalar).collect();
    assert_eq!(approx(add_results, 6), approx(expected_add, 6));

    let sub_results = run_scalar(&input, sub_scalar::F32, scalar);
    let expected_sub: Vec<f32> = input.iter().map(|x| x - scalar).collect();
    assert_eq!(approx(sub_results, 6), approx(expected_sub, 6));

    let mul_results = run_scalar(&input, mul_scalar::F32, scalar);
    let expected_mul: Vec<f32> = input.iter().map(|x| x * scalar).collect();
    assert_eq!(approx(mul_results, 6), approx(expected_mul, 6));

    let div_results = run_scalar(&input, div_scalar::F32, scalar);
    let expected_div: Vec<f32> = input.iter().map(|x| x / scalar).collect();
    assert_eq!(approx(div_results, 6), approx(expected_div, 6));

    let pow_results = run_scalar(&input, pow_scalar::F32, scalar);
    let expected_pow: Vec<f32> = input.iter().map(|x| x.powf(scalar)).collect();
    assert_eq!(approx(pow_results, 6), approx(expected_pow, 6));
}

#[test]
fn unary_scalar_minmax_f32() {
    let input: Vec<f32> = vec![1.0f32, 2.0, 3.0];
    let scalar = 2.0f32;

    let maximum_results = run_scalar(&input, maximum_scalar::F32, scalar);
    let expected_maximum: Vec<f32> = input.iter().map(|x| x.max(scalar)).collect();
    assert_eq!(approx(maximum_results, 6), approx(expected_maximum, 6));

    let minimum_results = run_scalar(&input, minimum_scalar::F32, scalar);
    let expected_minimum: Vec<f32> = input.iter().map(|x| x.min(scalar)).collect();
    assert_eq!(approx(minimum_results, 6), approx(expected_minimum, 6));
}

// Parametric activations
#[test]
fn unary_leaky_relu_f32() {
    let input: Vec<f32> = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let alpha = 0.01f32;

    let results = run_scalar(&input, leaky_relu::F32, alpha);
    let expected: Vec<f32> = input.iter().map(|x| if *x > 0.0 { *x } else { alpha * x }).collect();
    assert_eq!(approx(results, 6), approx(expected, 6));
}

#[test]
fn unary_elu_f32() {
    let input: Vec<f32> = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let alpha = 1.0f32;

    let results = run_scalar(&input, elu::F32, alpha);
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

    let results = run_scalar(&input, prelu::F32, alpha);
    let expected: Vec<f32> = input.iter().map(|x| if *x > 0.0 { *x } else { alpha * x }).collect();
    assert_eq!(approx(results, 6), approx(expected, 6));
}
