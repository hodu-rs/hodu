use half::{bf16, f16};
use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_binary, Kernel, *},
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

fn run_binary<T: Clone>(x: &[T], y: &[T], name: Kernel) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;
    let left = new_buffer(&device, x);
    let right = new_buffer(&device, y);
    let output = device.new_buffer(std::mem::size_of_val(x), options).unwrap();

    let shape = vec![x.len()];
    let strides = vec![1];

    call_binary(
        &device,
        &command_buffer,
        &kernels,
        name,
        &shape,
        BufferOffset::zero_offset(&left),
        &strides,
        0,
        BufferOffset::zero_offset(&right),
        &strides,
        0,
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, x.len())
}

fn run_binary_cmp<T: Clone, O: Clone>(x: &[T], y: &[T], name: Kernel) -> Vec<O> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;
    let left = new_buffer(&device, x);
    let right = new_buffer(&device, y);
    let output = device.new_buffer(x.len() * std::mem::size_of::<O>(), options).unwrap();

    let shape = vec![x.len()];
    let strides = vec![1];

    call_binary(
        &device,
        &command_buffer,
        &kernels,
        name,
        &shape,
        BufferOffset::zero_offset(&left),
        &strides,
        0,
        BufferOffset::zero_offset(&right),
        &strides,
        0,
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, x.len())
}

#[test]
fn binary_add_f32() {
    let left = vec![1.0f32, 2.0, 3.0];
    let right = vec![2.0f32, 3.1, 4.2];
    let results = run_binary(&left, &right, add::F32);
    let expected: Vec<_> = left.iter().zip(right.iter()).map(|(&x, &y)| x + y).collect();
    assert_eq!(approx(results, 4), vec![3.0f32, 5.1, 7.2]);
    assert_eq!(approx(expected, 4), vec![3.0f32, 5.1, 7.2]);
}

#[test]
fn binary_ops_bf16() {
    let lhs: Vec<bf16> = [1.1f32, 2.2, 3.3].into_iter().map(bf16::from_f32).collect();
    let rhs: Vec<bf16> = [4.2f32, 5.5f32, 6.91f32].into_iter().map(bf16::from_f32).collect();

    macro_rules! binary_op {
        ($opname:ident, $opexpr:expr) => {{
            let results = run_binary(&lhs, &rhs, $opname::BF16);
            let expected: Vec<bf16> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&bf16, &bf16)| $opexpr(*x, *y))
                .collect();
            assert_eq!(results, expected);
        }};
    }

    binary_op!(add, |x, y| x + y);
    binary_op!(sub, |x, y| x - y);
    binary_op!(mul, |x, y| x * y);
    binary_op!(div, |x, y| x / y);
    binary_op!(minimum, |x: bf16, y| x.min(y));
    binary_op!(maximum, |x: bf16, y| x.max(y));
}

#[test]
fn binary_add_f16() {
    let left: Vec<f16> = [1.0f32, 2.0, 3.0].into_iter().map(f16::from_f32).collect();
    let right: Vec<f16> = [2.0f32, 3.1, 4.2].into_iter().map(f16::from_f32).collect();
    let results = run_binary(&left, &right, add::F16);
    assert_eq!(approx_f16(results, 1), vec![3.0f32, 5.1, 7.2]);
}

#[test]
fn binary_sub_f32() {
    let left = vec![5.0f32, 10.0, 15.0];
    let right = vec![2.0f32, 3.0, 5.0];
    let results = run_binary(&left, &right, sub::F32);
    assert_eq!(approx(results, 4), vec![3.0f32, 7.0, 10.0]);
}

#[test]
fn binary_mul_f32() {
    let left = vec![2.0f32, 3.0, 4.0];
    let right = vec![5.0f32, 6.0, 7.0];
    let results = run_binary(&left, &right, mul::F32);
    assert_eq!(approx(results, 4), vec![10.0f32, 18.0, 28.0]);
}

#[test]
fn binary_div_f32() {
    let left = vec![10.0f32, 20.0, 30.0];
    let right = vec![2.0f32, 4.0, 5.0];
    let results = run_binary(&left, &right, div::F32);
    assert_eq!(approx(results, 4), vec![5.0f32, 5.0, 6.0]);
}

#[test]
fn binary_maximum_f32() {
    let left = vec![1.0f32, 5.0, 3.0];
    let right = vec![2.0f32, 4.0, 6.0];
    let results = run_binary(&left, &right, maximum::F32);
    assert_eq!(approx(results, 4), vec![2.0f32, 5.0, 6.0]);
}

#[test]
fn binary_minimum_f32() {
    let left = vec![1.0f32, 5.0, 3.0];
    let right = vec![2.0f32, 4.0, 6.0];
    let results = run_binary(&left, &right, minimum::F32);
    assert_eq!(approx(results, 4), vec![1.0f32, 4.0, 3.0]);
}

#[test]
fn binary_eq_f32() {
    let left = vec![1.0f32, 2.0, 3.0, 4.0];
    let right = vec![1.0f32, 3.0, 3.0, 5.0];
    let results: Vec<u8> = run_binary_cmp(&left, &right, eq::F32);
    assert_eq!(results, vec![1u8, 0, 1, 0]);
}

#[test]
fn binary_ne_f32() {
    let left = vec![1.0f32, 2.0, 3.0, 4.0];
    let right = vec![1.0f32, 3.0, 3.0, 5.0];
    let results: Vec<u8> = run_binary_cmp(&left, &right, ne::F32);
    assert_eq!(results, vec![0u8, 1, 0, 1]);
}

#[test]
fn binary_lt_f32() {
    let left = vec![1.0f32, 2.0, 3.0, 4.0];
    let right = vec![2.0f32, 2.0, 2.0, 5.0];
    let results: Vec<u8> = run_binary_cmp(&left, &right, lt::F32);
    assert_eq!(results, vec![1u8, 0, 0, 1]);
}

#[test]
fn binary_le_f32() {
    let left = vec![1.0f32, 2.0, 3.0, 4.0];
    let right = vec![2.0f32, 2.0, 2.0, 5.0];
    let results: Vec<u8> = run_binary_cmp(&left, &right, le::F32);
    assert_eq!(results, vec![1u8, 1, 0, 1]);
}

#[test]
fn binary_gt_f32() {
    let left = vec![1.0f32, 2.0, 3.0, 4.0];
    let right = vec![2.0f32, 2.0, 2.0, 5.0];
    let results: Vec<u8> = run_binary_cmp(&left, &right, gt::F32);
    assert_eq!(results, vec![0u8, 0, 1, 0]);
}

#[test]
fn binary_ge_f32() {
    let left = vec![1.0f32, 2.0, 3.0, 4.0];
    let right = vec![2.0f32, 2.0, 2.0, 5.0];
    let results: Vec<u8> = run_binary_cmp(&left, &right, ge::F32);
    assert_eq!(results, vec![0u8, 1, 1, 0]);
}

#[test]
fn binary_logical_and_u8() {
    let left = vec![0u8, 0, 1, 1];
    let right = vec![0u8, 1, 0, 1];
    let results: Vec<u8> = run_binary(&left, &right, logical_and::U8);
    assert_eq!(results, vec![0u8, 0, 0, 1]);
}

#[test]
fn binary_logical_or_u8() {
    let left = vec![0u8, 0, 1, 1];
    let right = vec![0u8, 1, 0, 1];
    let results: Vec<u8> = run_binary(&left, &right, logical_or::U8);
    assert_eq!(results, vec![0u8, 1, 1, 1]);
}

#[test]
fn binary_logical_xor_u8() {
    let left = vec![0u8, 0, 1, 1];
    let right = vec![0u8, 1, 0, 1];
    let results: Vec<u8> = run_binary(&left, &right, logical_xor::U8);
    assert_eq!(results, vec![0u8, 1, 1, 0]);
}
