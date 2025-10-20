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

fn run_binary_logical<T: Clone, O: Clone>(x: &[T], y: &[T], name: Kernel) -> Vec<O> {
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
fn binary_ops_bool() {
    let lhs: Vec<bool> = vec![false, true, false, true];
    let rhs: Vec<bool> = vec![false, true, true, false];

    macro_rules! binary_op {
        ($opname:ident, $opexpr:expr) => {{
            let results = run_binary(&lhs, &rhs, $opname::BOOL);
            let expected: Vec<bool> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&bool, &bool)| $opexpr(*x, *y))
                .collect();
            assert_eq!(results, expected);
        }};
    }

    binary_op!(add, |x, y| x || y);
    binary_op!(sub, |x, y| x ^ y);
    binary_op!(mul, |x, y| x && y);
    binary_op!(div, |x, y| x && y);
    binary_op!(minimum, |x: bool, y| x && y);
    binary_op!(maximum, |x: bool, y| x || y);
}

#[test]
fn binary_ops_bf16() {
    let lhs: Vec<bf16> = vec![1.1f32, 2.2, 3.3].into_iter().map(bf16::from_f32).collect();
    let rhs: Vec<bf16> = vec![4.2f32, 5.5f32, 6.91f32].into_iter().map(bf16::from_f32).collect();

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
fn binary_ops_f16() {
    let lhs: Vec<f16> = vec![1.1f32, 2.2, 3.3].into_iter().map(f16::from_f32).collect();
    let rhs: Vec<f16> = vec![4.2f32, 5.5f32, 6.91f32].into_iter().map(f16::from_f32).collect();

    macro_rules! binary_op {
        ($opname:ident, $opexpr:expr) => {{
            let results = run_binary(&lhs, &rhs, $opname::F16);
            let expected: Vec<f16> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&f16, &f16)| $opexpr(*x, *y))
                .collect();
            assert_eq!(results, expected);
        }};
    }

    binary_op!(add, |x, y| x + y);
    binary_op!(sub, |x, y| x - y);
    binary_op!(mul, |x, y| x * y);
    binary_op!(div, |x, y| x / y);
    binary_op!(minimum, |x: f16, y| x.min(y));
    binary_op!(maximum, |x: f16, y| x.max(y));
}

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
    binary_op!(minimum, |x: f32, y| x.min(y));
    binary_op!(maximum, |x: f32, y| x.max(y));
}

#[test]
fn binary_logical_ops_bool() {
    let lhs: Vec<bool> = vec![true, false, true, false];
    let rhs: Vec<bool> = vec![true, true, false, true];

    macro_rules! binary_logical_op {
        ($opname:ident, $opexpr:expr) => {{
            let results: Vec<bool> = run_binary_logical(&lhs, &rhs, $opname::BOOL);
            let expected: Vec<bool> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&bool, &bool)| $opexpr(*x, *y))
                .collect();
            assert_eq!(results, expected);
        }};
    }

    binary_logical_op!(logical_and, |x, y| x && y);
    binary_logical_op!(logical_or, |x, y| x || y);
    binary_logical_op!(logical_xor, |x, y| x ^ y);
}

#[test]
fn binary_logical_ops_bf16() {
    let lhs: Vec<bf16> = vec![1.0f32, 0.0, 1.0, 0.0].into_iter().map(bf16::from_f32).collect();
    let rhs: Vec<bf16> = vec![1.0f32, 1.0, 0.0, 1.0].into_iter().map(bf16::from_f32).collect();

    macro_rules! binary_logical_op {
        ($opname:ident, $opexpr:expr) => {{
            let results: Vec<bool> = run_binary_logical(&lhs, &rhs, $opname::BF16);
            let expected: Vec<bool> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&bf16, &bf16)| $opexpr(*x, *y))
                .collect();
            assert_eq!(results, expected);
        }};
    }

    binary_logical_op!(logical_and, |x, y| (x != bf16::from_f32(0.0))
        && (y != bf16::from_f32(0.0)));
    binary_logical_op!(logical_or, |x, y| (x != bf16::from_f32(0.0))
        || (y != bf16::from_f32(0.0)));
    binary_logical_op!(logical_xor, |x, y| (x != bf16::from_f32(0.0))
        ^ (y != bf16::from_f32(0.0)));
}

#[test]
fn binary_logical_ops_f16() {
    let lhs: Vec<f16> = vec![1.0f32, 0.0, 1.0, 0.0].into_iter().map(f16::from_f32).collect();
    let rhs: Vec<f16> = vec![1.0f32, 1.0, 0.0, 1.0].into_iter().map(f16::from_f32).collect();

    macro_rules! binary_logical_op {
        ($opname:ident, $opexpr:expr) => {{
            let results: Vec<bool> = run_binary_logical(&lhs, &rhs, $opname::F16);
            let expected: Vec<bool> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&f16, &f16)| $opexpr(*x, *y))
                .collect();
            assert_eq!(results, expected);
        }};
    }

    binary_logical_op!(logical_and, |x, y| (x != f16::from_f32(0.0))
        && (y != f16::from_f32(0.0)));
    binary_logical_op!(logical_or, |x, y| (x != f16::from_f32(0.0))
        || (y != f16::from_f32(0.0)));
    binary_logical_op!(logical_xor, |x, y| (x != f16::from_f32(0.0))
        ^ (y != f16::from_f32(0.0)));
}

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

#[test]
fn cmp_ops_bool() {
    let lhs: Vec<bool> = vec![true, false, true, false];
    let rhs: Vec<bool> = vec![true, true, false, true];

    macro_rules! cmp_op {
        ($opname:ident, $opexpr:expr) => {{
            let results: Vec<bool> = run_binary_logical(&lhs, &rhs, $opname::BOOL);
            let expected: Vec<bool> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&bool, &bool)| $opexpr(*x, *y))
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

#[test]
fn cmp_ops_bf16() {
    let lhs: Vec<bf16> = vec![1.0f32, 2.0, 3.0, 4.0].into_iter().map(bf16::from_f32).collect();
    let rhs: Vec<bf16> = vec![1.0f32, 3.0, 3.0, 5.0].into_iter().map(bf16::from_f32).collect();

    macro_rules! cmp_op {
        ($opname:ident, $opexpr:expr) => {{
            let results: Vec<bool> = run_binary_logical(&lhs, &rhs, $opname::BF16);
            let expected: Vec<bool> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&bf16, &bf16)| $opexpr(*x, *y))
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

#[test]
fn cmp_ops_f16() {
    let lhs: Vec<f16> = vec![1.0f32, 2.0, 3.0, 4.0].into_iter().map(f16::from_f32).collect();
    let rhs: Vec<f16> = vec![1.0f32, 3.0, 3.0, 5.0].into_iter().map(f16::from_f32).collect();

    macro_rules! cmp_op {
        ($opname:ident, $opexpr:expr) => {{
            let results: Vec<bool> = run_binary_logical(&lhs, &rhs, $opname::F16);
            let expected: Vec<bool> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&f16, &f16)| $opexpr(*x, *y))
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
