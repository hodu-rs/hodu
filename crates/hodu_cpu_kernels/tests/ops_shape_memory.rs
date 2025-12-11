use hodu_cpu_kernels::*;

#[test]
fn test_flip_f32_1d_dim0() {
    // Input: [1, 2, 3, 4]
    // Flip dim 0
    // Output: [4, 3, 2, 1]
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let mut output = vec![0.0f32; 4];

    let num_els = 4;
    let num_dims = 1;
    let shape = vec![4];
    let flip_mask = vec![1]; // flip dim 0

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_flip_f32_1d_no_flip() {
    // Input: [1, 2, 3, 4]
    // No flip
    // Output: [1, 2, 3, 4]
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let mut output = vec![0.0f32; 4];

    let num_els = 4;
    let num_dims = 1;
    let shape = vec![4];
    let flip_mask = vec![0]; // no flip

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_flip_f32_2d_dim0() {
    // Input: [[1, 2, 3], [4, 5, 6]]  (2x3)
    // Flip dim 0
    // Output: [[4, 5, 6], [1, 2, 3]]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = vec![0.0f32; 6];

    let num_els = 6;
    let num_dims = 2;
    let shape = vec![2, 3];
    let flip_mask = vec![1, 0]; // flip dim 0 only

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_flip_f32_2d_dim1() {
    // Input: [[1, 2, 3], [4, 5, 6]]  (2x3)
    // Flip dim 1
    // Output: [[3, 2, 1], [6, 5, 4]]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = vec![0.0f32; 6];

    let num_els = 6;
    let num_dims = 2;
    let shape = vec![2, 3];
    let flip_mask = vec![0, 1]; // flip dim 1 only

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
}

#[test]
fn test_flip_f32_2d_both_dims() {
    // Input: [[1, 2, 3], [4, 5, 6]]  (2x3)
    // Flip both dims
    // Output: [[6, 5, 4], [3, 2, 1]]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = vec![0.0f32; 6];

    let num_els = 6;
    let num_dims = 2;
    let shape = vec![2, 3];
    let flip_mask = vec![1, 1]; // flip both dims

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_flip_i32_1d() {
    let input = [10i32, 20, 30, 40];
    let mut output = vec![0i32; 4];

    let num_els = 4;
    let num_dims = 1;
    let shape = vec![4];
    let flip_mask = vec![1];

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::I32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![40, 30, 20, 10]);
}

#[test]
fn test_flip_f32_3d() {
    // Input: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  (2x2x2)
    // Flip dim 2
    // Output: [[[2, 1], [4, 3]], [[6, 5], [8, 7]]]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0f32; 8];

    let num_els = 8;
    let num_dims = 3;
    let shape = vec![2, 2, 2];
    let flip_mask = vec![0, 0, 1]; // flip dim 2 only

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]);
}

#[test]
fn test_flip_u8() {
    let input = [1u8, 2, 3, 4, 5];
    let mut output = vec![0u8; 5];

    let num_els = 5;
    let num_dims = 1;
    let shape = vec![5];
    let flip_mask = vec![1];

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::U8,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![5, 4, 3, 2, 1]);
}
