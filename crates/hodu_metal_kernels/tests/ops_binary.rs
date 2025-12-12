#![allow(clippy::bool_comparison)]

use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_ops_binary, Kernel, *},
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

fn run_binary<T: Clone>(lhs: &[T], rhs: &[T], kernel: Kernel) -> Vec<T> {
    assert_eq!(lhs.len(), rhs.len());
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;
    let left = new_buffer(&device, lhs);
    let right = new_buffer(&device, rhs);
    let output = device.new_buffer(std::mem::size_of_val(lhs), options).unwrap();

    let shape = vec![lhs.len()];
    let strides = vec![1];
    let num_els = lhs.len();
    let num_dims = shape.len();

    let mut metadata = Vec::with_capacity(2 + num_dims * 4 + 2);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.extend(&strides);
    metadata.push(0); // lhs offset
    metadata.push(0); // rhs offset

    call_ops_binary(
        kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&left),
        BufferOffset::zero_offset(&right),
        &output,
        &metadata,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, lhs.len())
}

fn run_binary_logical<T: Clone, O: Clone>(lhs: &[T], rhs: &[T], kernel: Kernel) -> Vec<O> {
    assert_eq!(lhs.len(), rhs.len());
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;
    let left = new_buffer(&device, lhs);
    let right = new_buffer(&device, rhs);
    let output = device
        .new_buffer(lhs.len() * std::mem::size_of::<O>(), options)
        .unwrap();

    let shape = vec![lhs.len()];
    let strides = vec![1];
    let num_els = lhs.len();
    let num_dims = shape.len();

    let mut metadata = Vec::with_capacity(2 + num_dims * 4 + 2);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.extend(&strides);
    metadata.push(0); // lhs offset
    metadata.push(0); // rhs offset

    call_ops_binary(
        kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&left),
        BufferOffset::zero_offset(&right),
        &output,
        &metadata,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, lhs.len())
}

// Arithmetic operations
#[test]
fn binary_ops_f32() {
    let lhs: Vec<f32> = vec![1.1f32, 2.2, 3.3];
    let rhs: Vec<f32> = vec![4.2f32, 5.5f32, 6.91f32];

    macro_rules! binary_op {
        ($opname:ident, $opexpr:expr) => {{
            let results = run_binary(&lhs, &rhs, $opname::F32);
            let expected: Vec<f32> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&f32, &f32)| $opexpr(*x, *y))
                .collect();
            assert_eq!(approx(results, 6), approx(expected, 6));
        }};
    }

    binary_op!(add, |x, y| x + y);
    binary_op!(sub, |x, y| x - y);
    binary_op!(mul, |x, y| x * y);
    binary_op!(div, |x, y| x / y);
    binary_op!(rem, |x: f32, y| x % y);
    binary_op!(minimum, |x: f32, y| x.min(y));
    binary_op!(maximum, |x: f32, y| x.max(y));
}

#[test]
fn binary_pow_f32() {
    let lhs: Vec<f32> = vec![2.0f32, 3.0, 4.0];
    let rhs: Vec<f32> = vec![2.0f32, 3.0, 0.5];

    let results = run_binary(&lhs, &rhs, pow::F32);
    let expected: Vec<f32> = lhs.iter().zip(rhs.iter()).map(|(x, y)| x.powf(*y)).collect();
    assert_eq!(approx(results, 4), approx(expected, 4));
}

// Logical operations
#[test]
fn binary_logical_ops_f32() {
    let lhs: Vec<f32> = vec![1.0f32, 0.0, 1.0, 0.0];
    let rhs: Vec<f32> = vec![1.0f32, 1.0, 0.0, 1.0];

    macro_rules! binary_logical_op {
        ($opname:ident, $opexpr:expr) => {{
            let results: Vec<bool> = run_binary_logical(&lhs, &rhs, $opname::F32);
            let expected: Vec<bool> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&f32, &f32)| $opexpr(*x, *y))
                .collect();
            assert_eq!(results, expected);
        }};
    }

    binary_logical_op!(logical_and, |x, y| (x != 0.0) && (y != 0.0));
    binary_logical_op!(logical_or, |x, y| (x != 0.0) || (y != 0.0));
    binary_logical_op!(logical_xor, |x, y| (x != 0.0) ^ (y != 0.0));
}

// Comparison operations
#[test]
fn cmp_ops_f32() {
    let lhs: Vec<f32> = vec![1.0f32, 2.0, 3.0, 4.0];
    let rhs: Vec<f32> = vec![1.0f32, 3.0, 3.0, 5.0];

    macro_rules! cmp_op {
        ($opname:ident, $opexpr:expr) => {{
            let results: Vec<bool> = run_binary_logical(&lhs, &rhs, $opname::F32);
            let expected: Vec<bool> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&f32, &f32)| $opexpr(*x, *y))
                .collect();
            assert_eq!(results, expected);
        }};
    }

    cmp_op!(eq, |x, y| x == y);
    cmp_op!(ne, |x, y| x != y);
    cmp_op!(lt, |x, y| x < y);
    cmp_op!(le, |x, y| x <= y);
    cmp_op!(gt, |x, y| x > y);
    cmp_op!(ge, |x, y| x >= y);
}
