use hodu_cpu_kernels::*;

#[test]
fn test_pad_constant_f32_1d() {
    // Input: [1, 2, 3]
    // Pad: 2 before, 1 after
    // Output: [0, 0, 1, 2, 3, 0]
    let input = [1.0f32, 2.0, 3.0];
    let pad_value = 0.0f32;
    let mut output = vec![0.0f32; 6];

    let num_dims = 1;
    let input_shape = vec![3];
    let output_shape = vec![6];
    let pad_before = vec![2];
    let num_els = 6;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_constant(
        pad_constant::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &pad_value as *const f32 as *const core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
}

#[test]
fn test_pad_constant_f32_1d_nonzero_value() {
    // Input: [1, 2, 3]
    // Pad: 1 before, 2 after with value -1
    // Output: [-1, 1, 2, 3, -1, -1]
    let input = [1.0f32, 2.0, 3.0];
    let pad_value = -1.0f32;
    let mut output = vec![0.0f32; 6];

    let num_dims = 1;
    let input_shape = vec![3];
    let output_shape = vec![6];
    let pad_before = vec![1];
    let num_els = 6;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_constant(
        pad_constant::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &pad_value as *const f32 as *const core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![-1.0, 1.0, 2.0, 3.0, -1.0, -1.0]);
}

#[test]
fn test_pad_constant_f32_2d() {
    // Input: [[1, 2], [3, 4]]  (2x2)
    // Pad: (1, 1) before, (1, 1) after
    // Output: [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]  (4x4)
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let pad_value = 0.0f32;
    let mut output = vec![0.0f32; 16];

    let num_dims = 2;
    let input_shape = vec![2, 2];
    let output_shape = vec![4, 4];
    let pad_before = vec![1, 1];
    let num_els = 16;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_constant(
        pad_constant::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &pad_value as *const f32 as *const core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    #[rustfmt::skip]
    let expected = vec![
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 2.0, 0.0,
        0.0, 3.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(output, expected);
}

#[test]
fn test_pad_constant_i32() {
    let input = [10i32, 20, 30];
    let pad_value = -99i32;
    let mut output = vec![0i32; 5];

    let num_dims = 1;
    let input_shape = vec![3];
    let output_shape = vec![5];
    let pad_before = vec![1];
    let num_els = 5;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_constant(
        pad_constant::I32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &pad_value as *const i32 as *const core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![-99, 10, 20, 30, -99]);
}

#[test]
fn test_pad_reflect_f32_1d() {
    // Input: [1, 2, 3, 4]
    // Pad: 2 before, 2 after
    // Output: [3, 2, 1, 2, 3, 4, 3, 2]
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let mut output = vec![0.0f32; 8];

    let num_dims = 1;
    let input_shape = vec![4];
    let output_shape = vec![8];
    let pad_before = vec![2];
    let num_els = 8;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_reflect(
        pad_reflect::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
}

#[test]
fn test_pad_reflect_f32_2d() {
    // Input: [[1, 2, 3], [4, 5, 6]]  (2x3)
    // Pad: (1, 1) before, (1, 1) after on dim 1 only
    // Output: [[2, 1, 2, 3, 2], [5, 4, 5, 6, 5]]  (2x5)
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = vec![0.0f32; 10];

    let num_dims = 2;
    let input_shape = vec![2, 3];
    let output_shape = vec![2, 5];
    let pad_before = vec![0, 1];
    let num_els = 10;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_reflect(
        pad_reflect::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    #[rustfmt::skip]
    let expected = vec![
        2.0, 1.0, 2.0, 3.0, 2.0,
        5.0, 4.0, 5.0, 6.0, 5.0,
    ];
    assert_eq!(output, expected);
}

#[test]
fn test_pad_reflect_i32() {
    let input = [10i32, 20, 30];
    let mut output = vec![0i32; 7];

    let num_dims = 1;
    let input_shape = vec![3];
    let output_shape = vec![7];
    let pad_before = vec![2];
    let num_els = 7;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_reflect(
        pad_reflect::I32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![30, 20, 10, 20, 30, 20, 10]);
}

#[test]
fn test_pad_replicate_f32_1d() {
    // Input: [1, 2, 3]
    // Pad: 2 before, 2 after
    // Output: [1, 1, 1, 2, 3, 3, 3]
    let input = [1.0f32, 2.0, 3.0];
    let mut output = vec![0.0f32; 7];

    let num_dims = 1;
    let input_shape = vec![3];
    let output_shape = vec![7];
    let pad_before = vec![2];
    let num_els = 7;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_replicate(
        pad_replicate::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]);
}

#[test]
fn test_pad_replicate_f32_2d() {
    // Input: [[1, 2], [3, 4]]  (2x2)
    // Pad: (1, 1) before/after on both dims
    // Output: [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]  (4x4)
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let mut output = vec![0.0f32; 16];

    let num_dims = 2;
    let input_shape = vec![2, 2];
    let output_shape = vec![4, 4];
    let pad_before = vec![1, 1];
    let num_els = 16;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_replicate(
        pad_replicate::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    #[rustfmt::skip]
    let expected = vec![
        1.0, 1.0, 2.0, 2.0,
        1.0, 1.0, 2.0, 2.0,
        3.0, 3.0, 4.0, 4.0,
        3.0, 3.0, 4.0, 4.0,
    ];
    assert_eq!(output, expected);
}

#[test]
fn test_pad_replicate_i32() {
    let input = [10i32, 20, 30];
    let mut output = vec![0i32; 6];

    let num_dims = 1;
    let input_shape = vec![3];
    let output_shape = vec![6];
    let pad_before = vec![1];
    let num_els = 6;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_replicate(
        pad_replicate::I32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![10, 10, 20, 30, 30, 30]);
}

#[test]
fn test_pad_circular_f32_1d() {
    // Input: [1, 2, 3]
    // Pad: 2 before, 2 after
    // Output: [2, 3, 1, 2, 3, 1, 2]
    let input = [1.0f32, 2.0, 3.0];
    let mut output = vec![0.0f32; 7];

    let num_dims = 1;
    let input_shape = vec![3];
    let output_shape = vec![7];
    let pad_before = vec![2];
    let num_els = 7;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_circular(
        pad_circular::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]);
}

#[test]
fn test_pad_circular_f32_2d() {
    // Input: [[1, 2], [3, 4]]  (2x2)
    // Pad: (1, 1) before/after on dim 1 only
    // Output: [[2, 1, 2, 1], [4, 3, 4, 3]]  (2x4)
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let mut output = vec![0.0f32; 8];

    let num_dims = 2;
    let input_shape = vec![2, 2];
    let output_shape = vec![2, 4];
    let pad_before = vec![0, 1];
    let num_els = 8;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_circular(
        pad_circular::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    #[rustfmt::skip]
    let expected = vec![
        2.0, 1.0, 2.0, 1.0,
        4.0, 3.0, 4.0, 3.0,
    ];
    assert_eq!(output, expected);
}

#[test]
fn test_pad_circular_i32() {
    let input = [10i32, 20, 30, 40];
    let mut output = vec![0i32; 8];

    let num_dims = 1;
    let input_shape = vec![4];
    let output_shape = vec![8];
    let pad_before = vec![2];
    let num_els = 8;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_circular(
        pad_circular::I32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![30, 40, 10, 20, 30, 40, 10, 20]);
}
