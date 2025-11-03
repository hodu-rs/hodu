use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};
use hodu_cpu_kernels::*;

fn run_binary_f8e4m3(lhs: &[F8E4M3], rhs: &[F8E4M3], kernel: Kernel) -> Vec<F8E4M3> {
    let mut output = vec![F8E4M3::ZERO; lhs.len()];
    let shape = vec![lhs.len()];
    let strides = vec![1];
    let num_els: usize = shape.iter().product();
    let num_dims = shape.len();
    let mut metadata = Vec::with_capacity(2 + num_dims * 4 + 1);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    call_binary(
        kernel,
        lhs.as_ptr() as *const core::ffi::c_void,
        rhs.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();
    output
}

fn run_binary_f8e5m2(lhs: &[F8E5M2], rhs: &[F8E5M2], kernel: Kernel) -> Vec<F8E5M2> {
    let mut output = vec![F8E5M2::ZERO; lhs.len()];
    let shape = vec![lhs.len()];
    let strides = vec![1];
    let num_els: usize = shape.iter().product();
    let num_dims = shape.len();
    let mut metadata = Vec::with_capacity(2 + num_dims * 4 + 1);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    call_binary(
        kernel,
        lhs.as_ptr() as *const core::ffi::c_void,
        rhs.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();
    output
}

fn run_binary_bf16(lhs: &[bf16], rhs: &[bf16], kernel: Kernel) -> Vec<bf16> {
    let mut output = vec![bf16::ZERO; lhs.len()];
    let shape = vec![lhs.len()];
    let strides = vec![1];
    let num_els: usize = shape.iter().product();
    let num_dims = shape.len();
    let mut metadata = Vec::with_capacity(2 + num_dims * 4 + 1);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    call_binary(
        kernel,
        lhs.as_ptr() as *const core::ffi::c_void,
        rhs.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();
    output
}

fn run_binary_f16(lhs: &[f16], rhs: &[f16], kernel: Kernel) -> Vec<f16> {
    let mut output = vec![f16::ZERO; lhs.len()];
    let shape = vec![lhs.len()];
    let strides = vec![1];
    let num_els: usize = shape.iter().product();
    let num_dims = shape.len();
    let mut metadata = Vec::with_capacity(2 + num_dims * 4 + 1);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    call_binary(
        kernel,
        lhs.as_ptr() as *const core::ffi::c_void,
        rhs.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();
    output
}

#[test]
fn test_add_f8e4m3() {
    let lhs = vec![
        F8E4M3::from_f32(1.0),
        F8E4M3::from_f32(2.0),
        F8E4M3::from_f32(3.0),
        F8E4M3::from_f32(4.0),
    ];
    let rhs = vec![
        F8E4M3::from_f32(0.5),
        F8E4M3::from_f32(1.0),
        F8E4M3::from_f32(1.5),
        F8E4M3::from_f32(2.0),
    ];

    let result = run_binary_f8e4m3(&lhs, &rhs, add::F8E4M3);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_mul_f8e4m3() {
    let lhs = vec![
        F8E4M3::from_f32(2.0),
        F8E4M3::from_f32(3.0),
        F8E4M3::from_f32(4.0),
        F8E4M3::from_f32(5.0),
    ];
    let rhs = vec![
        F8E4M3::from_f32(2.0),
        F8E4M3::from_f32(2.0),
        F8E4M3::from_f32(2.0),
        F8E4M3::from_f32(2.0),
    ];

    let result = run_binary_f8e4m3(&lhs, &rhs, mul::F8E4M3);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_add_f8e5m2() {
    let lhs = vec![
        F8E5M2::from_f32(1.0),
        F8E5M2::from_f32(2.0),
        F8E5M2::from_f32(3.0),
        F8E5M2::from_f32(4.0),
    ];
    let rhs = vec![
        F8E5M2::from_f32(0.5),
        F8E5M2::from_f32(1.0),
        F8E5M2::from_f32(1.5),
        F8E5M2::from_f32(2.0),
    ];

    let result = run_binary_f8e5m2(&lhs, &rhs, add::F8E5M2);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_mul_f8e5m2() {
    let lhs = vec![
        F8E5M2::from_f32(2.0),
        F8E5M2::from_f32(3.0),
        F8E5M2::from_f32(4.0),
        F8E5M2::from_f32(5.0),
    ];
    let rhs = vec![
        F8E5M2::from_f32(2.0),
        F8E5M2::from_f32(2.0),
        F8E5M2::from_f32(2.0),
        F8E5M2::from_f32(2.0),
    ];

    let result = run_binary_f8e5m2(&lhs, &rhs, mul::F8E5M2);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_add_bf16() {
    let lhs = vec![
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
        bf16::from_f32(4.0),
    ];
    let rhs = vec![
        bf16::from_f32(0.5),
        bf16::from_f32(1.0),
        bf16::from_f32(1.5),
        bf16::from_f32(2.0),
    ];

    let result = run_binary_bf16(&lhs, &rhs, add::BF16);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_mul_bf16() {
    let lhs = vec![
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
        bf16::from_f32(4.0),
        bf16::from_f32(5.0),
    ];
    let rhs = vec![
        bf16::from_f32(2.0),
        bf16::from_f32(2.0),
        bf16::from_f32(2.0),
        bf16::from_f32(2.0),
    ];

    let result = run_binary_bf16(&lhs, &rhs, mul::BF16);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_add_f16() {
    let lhs = vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ];
    let rhs = vec![
        f16::from_f32(0.5),
        f16::from_f32(1.0),
        f16::from_f32(1.5),
        f16::from_f32(2.0),
    ];

    let result = run_binary_f16(&lhs, &rhs, add::F16);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_mul_f16() {
    let lhs = vec![
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
        f16::from_f32(5.0),
    ];
    let rhs = vec![
        f16::from_f32(2.0),
        f16::from_f32(2.0),
        f16::from_f32(2.0),
        f16::from_f32(2.0),
    ];

    let result = run_binary_f16(&lhs, &rhs, mul::F16);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_sub_f8e4m3() {
    let lhs = vec![F8E4M3::from_f32(5.0), F8E4M3::from_f32(10.0), F8E4M3::from_f32(15.0)];
    let rhs = vec![F8E4M3::from_f32(2.0), F8E4M3::from_f32(3.0), F8E4M3::from_f32(5.0)];

    let result = run_binary_f8e4m3(&lhs, &rhs, sub::F8E4M3);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_div_bf16() {
    let lhs = vec![bf16::from_f32(10.0), bf16::from_f32(20.0), bf16::from_f32(30.0)];
    let rhs = vec![bf16::from_f32(2.0), bf16::from_f32(4.0), bf16::from_f32(5.0)];

    let result = run_binary_bf16(&lhs, &rhs, div::BF16);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_maximum_f16() {
    let lhs = vec![f16::from_f32(1.0), f16::from_f32(5.0), f16::from_f32(3.0)];
    let rhs = vec![f16::from_f32(2.0), f16::from_f32(4.0), f16::from_f32(6.0)];

    let result = run_binary_f16(&lhs, &rhs, maximum::F16);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_minimum_f8e5m2() {
    let lhs = vec![F8E5M2::from_f32(1.0), F8E5M2::from_f32(5.0), F8E5M2::from_f32(3.0)];
    let rhs = vec![F8E5M2::from_f32(2.0), F8E5M2::from_f32(4.0), F8E5M2::from_f32(6.0)];

    let result = run_binary_f8e5m2(&lhs, &rhs, minimum::F8E5M2);
    assert_eq!(result.len(), 3);
}
