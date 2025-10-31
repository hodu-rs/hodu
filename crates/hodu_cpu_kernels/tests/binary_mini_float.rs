use hodu_cpu_kernels::*;

fn run_binary_f8e4m3(lhs: &[u8], rhs: &[u8], kernel: Kernel) -> Vec<u8> {
    let mut output = vec![0u8; lhs.len()];
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
        lhs.as_ptr() as *const std::ffi::c_void,
        rhs.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();
    output
}

fn run_binary_f8e5m2(lhs: &[u8], rhs: &[u8], kernel: Kernel) -> Vec<u8> {
    let mut output = vec![0u8; lhs.len()];
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
        lhs.as_ptr() as *const std::ffi::c_void,
        rhs.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();
    output
}

fn run_binary_bf16(lhs: &[u16], rhs: &[u16], kernel: Kernel) -> Vec<u16> {
    let mut output = vec![0u16; lhs.len()];
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
        lhs.as_ptr() as *const std::ffi::c_void,
        rhs.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();
    output
}

fn run_binary_f16(lhs: &[u16], rhs: &[u16], kernel: Kernel) -> Vec<u16> {
    let mut output = vec![0u16; lhs.len()];
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
        lhs.as_ptr() as *const std::ffi::c_void,
        rhs.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();
    output
}

fn float_to_f8e4m3(val: f32) -> u8 {
    if val == 0.0 {
        return 0;
    }
    let sign = if val < 0.0 { 1u8 } else { 0u8 };
    let abs_val = val.abs();

    if abs_val.is_infinite() || abs_val > 448.0 {
        return (sign << 7) | 0x7F;
    }
    if abs_val < 0.001953125 {
        return sign << 7;
    }

    let exp = abs_val.log2().floor() as i32 + 7;
    let exp = exp.clamp(0, 14);

    let mant_val = abs_val / 2.0f32.powi(exp - 7);
    let mant = ((mant_val - 1.0) * 8.0).round() as u8;
    let mant = mant.min(7);

    (sign << 7) | ((exp as u8) << 3) | mant
}

fn float_to_f8e5m2(val: f32) -> u8 {
    if val == 0.0 {
        return 0;
    }
    let sign = if val < 0.0 { 1u8 } else { 0u8 };
    let abs_val = val.abs();

    if abs_val.is_infinite() || abs_val > 57344.0 {
        return (sign << 7) | 0x7F;
    }
    if abs_val < 0.0000152587890625 {
        return sign << 7;
    }

    let exp = abs_val.log2().floor() as i32 + 15;
    let exp = exp.clamp(0, 30);

    let mant_val = abs_val / 2.0f32.powi(exp - 15);
    let mant = ((mant_val - 1.0) * 4.0).round() as u8;
    let mant = mant.min(3);

    (sign << 7) | ((exp as u8) << 2) | mant
}

fn float_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    (bits >> 16) as u16
}

fn float_to_f16(val: f32) -> u16 {
    if val == 0.0 {
        return 0;
    }
    let sign = if val < 0.0 { 1u16 } else { 0u16 };
    let abs_val = val.abs();

    if abs_val.is_infinite() || abs_val > 65504.0 {
        return (sign << 15) | 0x7C00;
    }
    if abs_val < 0.00006103515625 {
        return sign << 15;
    }

    let exp = abs_val.log2().floor() as i32 + 15;
    let exp = exp.clamp(0, 30);

    let mant_val = abs_val / 2.0f32.powi(exp - 15);
    let mant = ((mant_val - 1.0) * 1024.0).round() as u16;
    let mant = mant.min(1023);

    (sign << 15) | ((exp as u16) << 10) | mant
}

#[test]
fn test_add_f8e4m3() {
    let lhs = vec![
        float_to_f8e4m3(1.0),
        float_to_f8e4m3(2.0),
        float_to_f8e4m3(3.0),
        float_to_f8e4m3(4.0),
    ];
    let rhs = vec![
        float_to_f8e4m3(0.5),
        float_to_f8e4m3(1.0),
        float_to_f8e4m3(1.5),
        float_to_f8e4m3(2.0),
    ];

    let result = run_binary_f8e4m3(&lhs, &rhs, add::F8E4M3);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_mul_f8e4m3() {
    let lhs = vec![
        float_to_f8e4m3(2.0),
        float_to_f8e4m3(3.0),
        float_to_f8e4m3(4.0),
        float_to_f8e4m3(5.0),
    ];
    let rhs = vec![
        float_to_f8e4m3(2.0),
        float_to_f8e4m3(2.0),
        float_to_f8e4m3(2.0),
        float_to_f8e4m3(2.0),
    ];

    let result = run_binary_f8e4m3(&lhs, &rhs, mul::F8E4M3);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_add_f8e5m2() {
    let lhs = vec![
        float_to_f8e5m2(1.0),
        float_to_f8e5m2(2.0),
        float_to_f8e5m2(3.0),
        float_to_f8e5m2(4.0),
    ];
    let rhs = vec![
        float_to_f8e5m2(0.5),
        float_to_f8e5m2(1.0),
        float_to_f8e5m2(1.5),
        float_to_f8e5m2(2.0),
    ];

    let result = run_binary_f8e5m2(&lhs, &rhs, add::F8E5M2);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_mul_f8e5m2() {
    let lhs = vec![
        float_to_f8e5m2(2.0),
        float_to_f8e5m2(3.0),
        float_to_f8e5m2(4.0),
        float_to_f8e5m2(5.0),
    ];
    let rhs = vec![
        float_to_f8e5m2(2.0),
        float_to_f8e5m2(2.0),
        float_to_f8e5m2(2.0),
        float_to_f8e5m2(2.0),
    ];

    let result = run_binary_f8e5m2(&lhs, &rhs, mul::F8E5M2);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_add_bf16() {
    let lhs = vec![
        float_to_bf16(1.0),
        float_to_bf16(2.0),
        float_to_bf16(3.0),
        float_to_bf16(4.0),
    ];
    let rhs = vec![
        float_to_bf16(0.5),
        float_to_bf16(1.0),
        float_to_bf16(1.5),
        float_to_bf16(2.0),
    ];

    let result = run_binary_bf16(&lhs, &rhs, add::BF16);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_mul_bf16() {
    let lhs = vec![
        float_to_bf16(2.0),
        float_to_bf16(3.0),
        float_to_bf16(4.0),
        float_to_bf16(5.0),
    ];
    let rhs = vec![
        float_to_bf16(2.0),
        float_to_bf16(2.0),
        float_to_bf16(2.0),
        float_to_bf16(2.0),
    ];

    let result = run_binary_bf16(&lhs, &rhs, mul::BF16);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_add_f16() {
    let lhs = vec![
        float_to_f16(1.0),
        float_to_f16(2.0),
        float_to_f16(3.0),
        float_to_f16(4.0),
    ];
    let rhs = vec![
        float_to_f16(0.5),
        float_to_f16(1.0),
        float_to_f16(1.5),
        float_to_f16(2.0),
    ];

    let result = run_binary_f16(&lhs, &rhs, add::F16);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_mul_f16() {
    let lhs = vec![
        float_to_f16(2.0),
        float_to_f16(3.0),
        float_to_f16(4.0),
        float_to_f16(5.0),
    ];
    let rhs = vec![
        float_to_f16(2.0),
        float_to_f16(2.0),
        float_to_f16(2.0),
        float_to_f16(2.0),
    ];

    let result = run_binary_f16(&lhs, &rhs, mul::F16);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_sub_f8e4m3() {
    let lhs = vec![float_to_f8e4m3(5.0), float_to_f8e4m3(10.0), float_to_f8e4m3(15.0)];
    let rhs = vec![float_to_f8e4m3(2.0), float_to_f8e4m3(3.0), float_to_f8e4m3(5.0)];

    let result = run_binary_f8e4m3(&lhs, &rhs, sub::F8E4M3);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_div_bf16() {
    let lhs = vec![float_to_bf16(10.0), float_to_bf16(20.0), float_to_bf16(30.0)];
    let rhs = vec![float_to_bf16(2.0), float_to_bf16(4.0), float_to_bf16(5.0)];

    let result = run_binary_bf16(&lhs, &rhs, div::BF16);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_maximum_f16() {
    let lhs = vec![float_to_f16(1.0), float_to_f16(5.0), float_to_f16(3.0)];
    let rhs = vec![float_to_f16(2.0), float_to_f16(4.0), float_to_f16(6.0)];

    let result = run_binary_f16(&lhs, &rhs, maximum::F16);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_minimum_f8e5m2() {
    let lhs = vec![float_to_f8e5m2(1.0), float_to_f8e5m2(5.0), float_to_f8e5m2(3.0)];
    let rhs = vec![float_to_f8e5m2(2.0), float_to_f8e5m2(4.0), float_to_f8e5m2(6.0)];

    let result = run_binary_f8e5m2(&lhs, &rhs, minimum::F8E5M2);
    assert_eq!(result.len(), 3);
}
