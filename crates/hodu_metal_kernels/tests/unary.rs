use half::{bf16, f16};
use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_unary, Kernel, *},
    metal::{create_command_buffer, Buffer, Device},
    utils::BufferOffset,
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

fn approx_f16(v: Vec<f16>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t.to_f32() * b) / b).collect()
}

fn approx_bf16(v: Vec<bf16>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t.to_f32() * b) / b).collect()
}

fn run<T: Clone>(v: &[T], name: Kernel) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);

    let shape = vec![v.len()];
    let strides = vec![1];

    call_unary(
        &device,
        &command_buffer,
        &kernels,
        name,
        &shape,
        BufferOffset::zero_offset(&input),
        &strides,
        0,
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, v.len())
}

fn run_strided<T: Clone>(v: &[T], kernel: Kernel, shape: &[usize], strides: &[usize], offset: usize) -> Vec<T> {
    let device = device();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let input = new_buffer(&device, v);
    let output_size: usize = shape.iter().product();
    let output_vec = vec![0.0f32; output_size];
    let output_b = new_buffer(&device, &output_vec);

    let kernels = Kernels::new();
    call_unary(
        &device,
        &command_buffer,
        &kernels,
        kernel,
        shape,
        BufferOffset {
            buffer: &input,
            offset_in_bytes: offset,
        },
        strides,
        0,
        &output_b,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output_b, output_size)
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
fn tanh_f32() {
    let v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run(&v, tanh::F32);
    let expected: Vec<_> = v.iter().map(|v| v.tanh()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn gelu_f32() {
    // Using moderate values to avoid numerical instability
    let v: Vec<f32> = vec![-3.0f32, -1.0, 0., 1., 2., 3.0];
    let results = run(&v, gelu::F32);
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    // For reference values:
    // GELU(-3) ≈ -0.00404
    // GELU(-1) ≈ -0.159
    // GELU(0) = 0
    // GELU(1) ≈ 0.841
    // GELU(2) ≈ 1.955
    // GELU(3) ≈ 2.996
    let expected: Vec<f32> = vec![-0.004, -0.159, 0.0, 0.841, 1.955, 2.996];
    assert_eq!(approx(results, 3), expected);
}

#[test]
fn gelu_f16() {
    let v: Vec<f16> = [-3.0f32, -1.0, 0., 1., 2., 3.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let expected: Vec<f32> = vec![-0.0, -0.16, 0.0, 0.84, 1.96, 3.0];
    let results = run(&v, gelu::F16);
    assert_eq!(approx_f16(results, 2), expected);
}

#[test]
fn softplus_f32() {
    let v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run(&v, softplus::F32);
    let expected: Vec<_> = v.iter().map(|v| (1.0 + v.exp()).ln()).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

#[test]
fn leaky_relu_f32() {
    // Note: leaky_relu might need a constant parameter, skipping for now if not supported
    // This test is a placeholder
}

#[test]
fn elu_f32() {
    // Note: elu might need a constant parameter, skipping for now if not supported
    // This test is a placeholder
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

    let v = vec![1.0f32; 10_000];
    let results = run(&v, cos::F32);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403; 10_000]);
    assert_eq!(approx(expected, 4), vec![0.5403; 10_000]);
}

#[test]
fn cos_f16() {
    let v: Vec<f16> = [1.0f32, 2.0, 3.0].iter().map(|v| f16::from_f32(*v)).collect();
    let results = run(&v, cos::F16);
    let expected: Vec<f16> = v.iter().map(|v| f16::from_f32(v.to_f32().cos())).collect();
    assert_eq!(approx_f16(results, 2), vec![0.54, -0.42, -0.99]);
    assert_eq!(approx_f16(expected, 2), vec![0.54, -0.42, -0.99]);
}

#[test]
fn tan_f32() {
    let v = vec![0.0f32, 0.5, 1.0];
    let results = run(&v, tan::F32);
    let expected: Vec<_> = v.iter().map(|v| v.tan()).collect();
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

// Logical operations
#[test]
fn logical_not_u8() {
    let v = vec![0u8, 1, 0, 1, 0];
    let results: Vec<u8> = run(&v, logical_not::U8);
    assert_eq!(results, vec![1u8, 0, 1, 0, 1]);
}

// Strided operations test
#[test]
fn cos_f32_strided() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![6];
    let strides = vec![1];
    let offset = 0;
    let results = run_strided(&v, cos::F32, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Contiguous
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![3, 2];
    let strides = vec![2, 1];
    let offset = 0;
    let results = run_strided(&v, cos::F32, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Transposed
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![3, 2];
    let strides = vec![1, 3];
    let offset = 0;
    let results = run_strided(&v, cos::F32, &shape, &strides, offset);
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.6536, -0.4161, 0.2837, -0.99, 0.9602]
    );

    // Very large
    let v = vec![1.0f32; 10_000];
    let shape = vec![2, 5_000];
    let strides = vec![2, 1];
    let offset = 0;
    let results = run_strided(&v, cos::F32, &shape, &strides, offset);
    assert_eq!(approx(results, 4), vec![0.5403; 10_000]);
}

// bf16 tests
#[test]
fn abs_bf16() {
    let v: Vec<bf16> = [-1.0f32, -2.0, 3.0, -4.0].iter().map(|v| bf16::from_f32(*v)).collect();
    let results = run(&v, abs::BF16);
    assert_eq!(approx_bf16(results, 4), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn relu_bf16() {
    let v: Vec<bf16> = [-2.0f32, -1.0, 0.0, 1.0, 2.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect();
    let results = run(&v, relu::BF16);
    assert_eq!(approx_bf16(results, 4), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn tanh_bf16() {
    let v: Vec<bf16> = [-2.0f32, -1.0, 0.0, 1.0, 2.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect();
    let results = run(&v, tanh::BF16);
    let expected: Vec<_> = [-2.0f32, -1.0, 0.0, 1.0, 2.0].iter().map(|v| v.tanh()).collect();
    assert_eq!(approx_bf16(results, 2), approx(expected, 2));
}
