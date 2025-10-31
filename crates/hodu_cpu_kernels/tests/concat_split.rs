use hodu_cpu_kernels::*;

#[test]
fn test_concat_f32_dim0() {
    // Two 2x2 matrices concatenated along dim 0 to form 4x2
    // Input 1: [[1, 2], [3, 4]]
    // Input 2: [[5, 6], [7, 8]]
    // Output: [[1, 2], [3, 4], [5, 6], [7, 8]]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0f32; 8];

    let output_shape = vec![4, 2];
    let concat_dim = 0;

    // Input shapes for 2 inputs: [2, 2, 2, 2] (flattened)
    let input_shapes = vec![2, 2, 2, 2];
    // Input strides: [2, 1, 2, 1]
    let input_strides = vec![2, 1, 2, 1];
    // Input offsets: [0, 0] (both start at 0)
    let input_offsets = vec![0, 0];
    // Buffer offsets (in elements): [0, 4] (second matrix starts at index 4)
    let input_buffer_offsets = vec![0, 4];

    call_concat(
        concat::F32,
        &output_shape,
        concat_dim,
        &input_shapes,
        &input_strides,
        &input_offsets,
        &input_buffer_offsets,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    );

    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_concat_f32_dim1() {
    // Two 2x2 matrices concatenated along dim 1 to form 2x4
    // Input 1: [[1, 2], [3, 4]]
    // Input 2: [[5, 6], [7, 8]]
    // Output: [[1, 2, 5, 6], [3, 4, 7, 8]]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0f32; 8];

    let output_shape = vec![2, 4];
    let concat_dim = 1;

    // Input shapes for 2 inputs: [2, 2, 2, 2] (flattened)
    let input_shapes = vec![2, 2, 2, 2];
    // Input strides: [2, 1, 2, 1]
    let input_strides = vec![2, 1, 2, 1];
    // Input offsets: [0, 0]
    let input_offsets = vec![0, 0];
    // Buffer offsets (in elements): [0, 4] (second matrix starts at index 4)
    let input_buffer_offsets = vec![0, 4];

    call_concat(
        concat::F32,
        &output_shape,
        concat_dim,
        &input_shapes,
        &input_strides,
        &input_offsets,
        &input_buffer_offsets,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    );

    assert_eq!(output, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn test_concat_i32_dim0() {
    // Test with integer type
    // Two 2x2 matrices concatenated along dim 0 to form 4x2
    let input = [1i32, 2, 3, 4, 5, 6, 7, 8];
    let mut output = vec![0i32; 8];

    let output_shape = vec![4, 2];
    let concat_dim = 0;

    let input_shapes = vec![2, 2, 2, 2];
    let input_strides = vec![2, 1, 2, 1];
    let input_offsets = vec![0, 0];
    let input_buffer_offsets = vec![0, 4];

    call_concat(
        concat::I32,
        &output_shape,
        concat_dim,
        &input_shapes,
        &input_strides,
        &input_offsets,
        &input_buffer_offsets,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    );

    assert_eq!(output, vec![1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn test_concat_3_tensors() {
    // Three 2x2 matrices concatenated along dim 0 to form 6x2
    // Input 1: [[1, 2], [3, 4]]
    // Input 2: [[5, 6], [7, 8]]
    // Input 3: [[9, 10], [11, 12]]
    // Output: [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut output = vec![0.0f32; 12];

    let output_shape = vec![6, 2];
    let concat_dim = 0;

    // Input shapes for 3 inputs: [2, 2, 2, 2, 2, 2] (flattened)
    let input_shapes = vec![2, 2, 2, 2, 2, 2];
    // Input strides: [2, 1, 2, 1, 2, 1]
    let input_strides = vec![2, 1, 2, 1, 2, 1];
    // Input offsets: [0, 0, 0]
    let input_offsets = vec![0, 0, 0];
    // Buffer offsets (in elements): [0, 4, 8]
    let input_buffer_offsets = vec![0, 4, 8];

    call_concat(
        concat::F32,
        &output_shape,
        concat_dim,
        &input_shapes,
        &input_strides,
        &input_offsets,
        &input_buffer_offsets,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    );

    assert_eq!(
        output,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );
}

#[test]
fn test_split_f32_dim0() {
    // Input: 4x2 matrix [[1, 2], [3, 4], [5, 6], [7, 8]]
    // Split along dim 0, taking first 2 rows: [[1, 2], [3, 4]]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0f32; 4];

    let input_shape = vec![4, 2];
    let strides = vec![2, 1];
    let split_dim = 0;
    let output_size_on_dim = 2; // Take 2 rows
    let split_offset = 0; // Start from beginning

    call_split(
        split::F32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &strides,
        0,
        split_dim,
        output_size_on_dim,
        split_offset,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    );

    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_split_f32_dim0_offset() {
    // Input: 4x2 matrix [[1, 2], [3, 4], [5, 6], [7, 8]]
    // Split along dim 0, taking 2 rows starting from offset 2: [[5, 6], [7, 8]]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0f32; 4];

    let input_shape = vec![4, 2];
    let strides = vec![2, 1];
    let split_dim = 0;
    let output_size_on_dim = 2;
    let split_offset = 2; // Start from row 2

    call_split(
        split::F32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &strides,
        0,
        split_dim,
        output_size_on_dim,
        split_offset,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    );

    assert_eq!(output, vec![5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_split_f32_dim1() {
    // Input: 2x4 matrix [[1, 2, 3, 4], [5, 6, 7, 8]]
    // Split along dim 1, taking first 2 columns: [[1, 2], [5, 6]]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0f32; 4];

    let input_shape = vec![2, 4];
    let strides = vec![4, 1];
    let split_dim = 1;
    let output_size_on_dim = 2; // Take 2 columns
    let split_offset = 0; // Start from beginning

    call_split(
        split::F32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &strides,
        0,
        split_dim,
        output_size_on_dim,
        split_offset,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    );

    assert_eq!(output, vec![1.0, 2.0, 5.0, 6.0]);
}

#[test]
fn test_split_f32_dim1_offset() {
    // Input: 2x4 matrix [[1, 2, 3, 4], [5, 6, 7, 8]]
    // Split along dim 1, taking 2 columns starting from offset 2: [[3, 4], [7, 8]]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0f32; 4];

    let input_shape = vec![2, 4];
    let strides = vec![4, 1];
    let split_dim = 1;
    let output_size_on_dim = 2;
    let split_offset = 2; // Start from column 2

    call_split(
        split::F32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &strides,
        0,
        split_dim,
        output_size_on_dim,
        split_offset,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    );

    assert_eq!(output, vec![3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn test_split_i32_dim0() {
    // Test with integer type
    // Input: 4x2 matrix, take first 2 rows
    let input = [1i32, 2, 3, 4, 5, 6, 7, 8];
    let mut output = vec![0i32; 4];

    let input_shape = vec![4, 2];
    let strides = vec![2, 1];
    let split_dim = 0;
    let output_size_on_dim = 2;
    let split_offset = 0;

    call_split(
        split::I32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &strides,
        0,
        split_dim,
        output_size_on_dim,
        split_offset,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    );

    assert_eq!(output, vec![1, 2, 3, 4]);
}

#[test]
fn test_split_3d_tensor() {
    // Input: 2x3x2 tensor
    // [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
    // Split along dim 1, taking first 2 rows from each batch
    // Output: 2x2x2 tensor [[[1, 2], [3, 4]], [[7, 8], [9, 10]]]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut output = vec![0.0f32; 8];

    let input_shape = vec![2, 3, 2];
    let strides = vec![6, 2, 1];
    let split_dim = 1;
    let output_size_on_dim = 2; // Take 2 rows
    let split_offset = 0;

    call_split(
        split::F32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &strides,
        0,
        split_dim,
        output_size_on_dim,
        split_offset,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    );

    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 10.0]);
}
