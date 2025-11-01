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

fn run<T: Clone>(v: &[T], name: Kernel) -> Vec<T> {
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

    call_unary(
        &device,
        &command_buffer,
        &kernels,
        name,
        BufferOffset::zero_offset(&input),
        &output,
        &metadata,
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
fn mish_f32() {
    let v = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let results = run(&v, mish::F32);
    let expected: Vec<_> = v.iter().map(|v| v * (1.0 + v.exp()).ln().tanh()).collect();
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
