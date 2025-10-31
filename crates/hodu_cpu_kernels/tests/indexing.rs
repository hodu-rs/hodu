use hodu_cpu_kernels::*;

#[test]
fn test_index_select_f32_1d() {
    // Input: [1, 2, 3, 4, 5]
    // Indices: [0, 2, 4] (select elements at positions 0, 2, 4)
    // Output: [1, 3, 5]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let indices = [0i32, 2, 4];
    let mut output = vec![0.0f32; 3];

    let shape = vec![5];
    let strides = vec![1];
    let input_offset = 0;
    let dim = 0;
    let num_indices = 3;

    call_index_select(
        index_select::F32,
        &shape,
        input.as_ptr() as *const std::ffi::c_void,
        &strides,
        input_offset,
        indices.as_ptr(),
        dim,
        num_indices,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![1.0, 3.0, 5.0]);
}

#[test]
fn test_index_select_f32_2d_dim0() {
    // Input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  (3x3)
    // Indices: [0, 2] (select rows 0 and 2)
    // Output: [[1, 2, 3], [7, 8, 9]]  (2x3)
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let indices = [0i32, 2];
    let mut output = vec![0.0f32; 6];

    let shape = vec![3, 3];
    let strides = vec![3, 1];
    let input_offset = 0;
    let dim = 0;
    let num_indices = 2;

    call_index_select(
        index_select::F32,
        &shape,
        input.as_ptr() as *const std::ffi::c_void,
        &strides,
        input_offset,
        indices.as_ptr(),
        dim,
        num_indices,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
}

#[test]
fn test_index_select_f32_2d_dim1() {
    // Input: [[1, 2, 3], [4, 5, 6]]  (2x3)
    // Indices: [0, 2] (select columns 0 and 2)
    // Output: [[1, 3], [4, 6]]  (2x2)
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices = [0i32, 2];
    let mut output = vec![0.0f32; 4];

    let shape = vec![2, 3];
    let strides = vec![3, 1];
    let input_offset = 0;
    let dim = 1;
    let num_indices = 2;

    call_index_select(
        index_select::F32,
        &shape,
        input.as_ptr() as *const std::ffi::c_void,
        &strides,
        input_offset,
        indices.as_ptr(),
        dim,
        num_indices,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![1.0, 3.0, 4.0, 6.0]);
}

#[test]
fn test_index_select_negative_indices() {
    // Input: [1, 2, 3, 4, 5]
    // Indices: [-1, -2] (select last two elements in reverse)
    // Output: [5, 4]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let indices = [-1i32, -2];
    let mut output = vec![0.0f32; 2];

    let shape = vec![5];
    let strides = vec![1];
    let input_offset = 0;
    let dim = 0;
    let num_indices = 2;

    call_index_select(
        index_select::F32,
        &shape,
        input.as_ptr() as *const std::ffi::c_void,
        &strides,
        input_offset,
        indices.as_ptr(),
        dim,
        num_indices,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![5.0, 4.0]);
}

#[test]
fn test_index_select_i32() {
    // Test with integer type
    let input = [10i32, 20, 30, 40, 50];
    let indices = [1i32, 3];
    let mut output = vec![0i32; 2];

    let shape = vec![5];
    let strides = vec![1];
    let input_offset = 0;
    let dim = 0;
    let num_indices = 2;

    call_index_select(
        index_select::I32,
        &shape,
        input.as_ptr() as *const std::ffi::c_void,
        &strides,
        input_offset,
        indices.as_ptr(),
        dim,
        num_indices,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![20, 40]);
}

#[test]
fn test_index_put_f32_1d() {
    // Input: [1, 2, 3, 4, 5]
    // Indices: [1, 3] (put at positions 1 and 3)
    // Values: [10, 20]
    // Output: [1, 10, 3, 20, 5]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let indices = [1i32, 3];
    let values = [10.0f32, 20.0];
    let mut output = vec![0.0f32; 5];

    let input_shape = vec![5];
    let input_strides = vec![1];
    let values_strides = vec![1];
    let input_offset = 0;
    let values_offset = 0;
    let dim = 0;
    let num_indices = 2;

    call_index_put(
        index_put::F32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &input_strides,
        input_offset,
        indices.as_ptr(),
        values.as_ptr() as *const std::ffi::c_void,
        &values_strides,
        values_offset,
        dim,
        num_indices,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![1.0, 10.0, 3.0, 20.0, 5.0]);
}

#[test]
fn test_index_put_f32_2d() {
    // Input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    // Indices: [0, 2] (put at rows 0 and 2)
    // Values: [[10, 11, 12], [13, 14, 15]]
    // Output: [[10, 11, 12], [4, 5, 6], [13, 14, 15]]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let indices = [0i32, 2];
    let values = [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0];
    let mut output = vec![0.0f32; 9];

    let input_shape = vec![3, 3];
    let input_strides = vec![3, 1];
    let values_strides = vec![3, 1];
    let input_offset = 0;
    let values_offset = 0;
    let dim = 0;
    let num_indices = 2;

    call_index_put(
        index_put::F32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &input_strides,
        input_offset,
        indices.as_ptr(),
        values.as_ptr() as *const std::ffi::c_void,
        &values_strides,
        values_offset,
        dim,
        num_indices,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![10.0, 11.0, 12.0, 4.0, 5.0, 6.0, 13.0, 14.0, 15.0]);
}

#[test]
fn test_gather_f32_1d() {
    // Input: [1, 2, 3, 4, 5]
    // Indices: [0, 2, 4, 1] (gather at these positions)
    // Output: [1, 3, 5, 2]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let indices = [0i32, 2, 4, 1];
    let mut output = vec![0.0f32; 4];

    let input_shape = vec![5];
    let input_strides = vec![1];
    let indices_strides = vec![1];
    let input_offset = 0;
    let indices_offset = 0;
    let dim = 0;

    call_gather(
        gather::F32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &input_strides,
        input_offset,
        indices.as_ptr(),
        &indices_strides,
        indices_offset,
        indices.len(),
        dim,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![1.0, 3.0, 5.0, 2.0]);
}

#[test]
fn test_gather_i32() {
    // Test with integer type - 1D case
    let input = [10i32, 20, 30, 40, 50];
    let indices = [4i32, 2, 0, 3];
    let mut output = vec![0i32; 4];

    let input_shape = vec![5];
    let input_strides = vec![1];
    let indices_strides = vec![1];
    let input_offset = 0;
    let indices_offset = 0;
    let dim = 0;

    call_gather(
        gather::I32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &input_strides,
        input_offset,
        indices.as_ptr(),
        &indices_strides,
        indices_offset,
        indices.len(),
        dim,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![50, 30, 10, 40]);
}

#[test]
fn test_gather_f32_2d() {
    // Input: [[1, 2, 3], [4, 5, 6]]  (2x3)
    // Indices: [0, 2, 1] (gather columns at positions 0, 2, 1 from dim 1)
    // Output: [[1, 3, 2], [4, 6, 5]]  (2x3)
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices = [0i32, 2, 1];
    let mut output = vec![0.0f32; 6];

    let input_shape = vec![2, 3];
    let input_strides = vec![3, 1];
    let indices_strides = vec![1];
    let input_offset = 0;
    let indices_offset = 0;
    let dim = 1;

    call_gather(
        gather::F32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &input_strides,
        input_offset,
        indices.as_ptr(),
        &indices_strides,
        indices_offset,
        indices.len(),
        dim,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![1.0, 3.0, 2.0, 4.0, 6.0, 5.0]);
}

#[test]
fn test_gather_f32_3d() {
    // Input: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  (2x2x2)
    // Indices: [1, 0] (gather from dim 2, selecting indices [1, 0])
    // Output: [[[2, 1], [4, 3]], [[6, 5], [8, 7]]]  (2x2x2)
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let indices = [1i32, 0];
    let mut output = vec![0.0f32; 8];

    let input_shape = vec![2, 2, 2];
    let input_strides = vec![4, 2, 1];
    let indices_strides = vec![1];
    let input_offset = 0;
    let indices_offset = 0;
    let dim = 2;

    call_gather(
        gather::F32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &input_strides,
        input_offset,
        indices.as_ptr(),
        &indices_strides,
        indices_offset,
        indices.len(),
        dim,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]);
}

#[test]
fn test_scatter_f32_1d() {
    // Input: [1, 2, 3, 4, 5]
    // Indices: [0, 2, 4]
    // Src: [10, 20, 30]
    // Output: [10, 2, 20, 4, 30]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let indices = [0i32, 2, 4];
    let src = [10.0f32, 20.0, 30.0];
    let mut output = vec![0.0f32; 5];

    let input_shape = vec![5];
    let input_strides = vec![1];
    let src_shape = vec![3];
    let src_strides = vec![1];
    let indices_strides = vec![1];
    let input_offset = 0;
    let src_offset = 0;
    let indices_offset = 0;
    let dim = 0;

    call_scatter(
        scatter::F32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &input_strides,
        input_offset,
        indices.as_ptr(),
        &indices_strides,
        indices_offset,
        src.as_ptr() as *const std::ffi::c_void,
        &src_shape,
        &src_strides,
        src_offset,
        dim,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![10.0, 2.0, 20.0, 4.0, 30.0]);
}

#[test]
fn test_scatter_f32_2d() {
    // Input: [[1, 2], [3, 4], [5, 6]]  (3x2)
    // Indices: [[0], [2]]  (2x1)
    // Src: [[10, 11], [12, 13]]  (2x2)
    // Output: [[10, 11], [3, 4], [12, 13]]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices = [0i32, 2];
    let src = [10.0f32, 11.0, 12.0, 13.0];
    let mut output = vec![0.0f32; 6];

    let input_shape = vec![3, 2];
    let input_strides = vec![2, 1];
    let src_shape = vec![2, 2];
    let src_strides = vec![2, 1];
    let indices_strides = vec![1];
    let input_offset = 0;
    let src_offset = 0;
    let indices_offset = 0;
    let dim = 0;

    call_scatter(
        scatter::F32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &input_strides,
        input_offset,
        indices.as_ptr(),
        &indices_strides,
        indices_offset,
        src.as_ptr() as *const std::ffi::c_void,
        &src_shape,
        &src_strides,
        src_offset,
        dim,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![10.0, 11.0, 3.0, 4.0, 12.0, 13.0]);
}

#[test]
fn test_scatter_add_f32_1d() {
    // Input: [1, 2, 3, 4, 5]
    // Indices: [0, 2, 0]  (note: index 0 appears twice)
    // Src: [10, 20, 30]
    // Output: [41, 2, 23, 4, 5]  (1+10+30=41, 3+20=23)
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let indices = [0i32, 2, 0];
    let src = [10.0f32, 20.0, 30.0];
    let mut output = vec![0.0f32; 5];

    let input_shape = vec![5];
    let input_strides = vec![1];
    let src_shape = vec![3];
    let src_strides = vec![1];
    let indices_strides = vec![1];
    let input_offset = 0;
    let src_offset = 0;
    let indices_offset = 0;
    let dim = 0;

    call_scatter(
        scatter_add::F32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &input_strides,
        input_offset,
        indices.as_ptr(),
        &indices_strides,
        indices_offset,
        src.as_ptr() as *const std::ffi::c_void,
        &src_shape,
        &src_strides,
        src_offset,
        dim,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![41.0, 2.0, 23.0, 4.0, 5.0]);
}

#[test]
fn test_scatter_add_i32() {
    // Test with integer type
    let input = [1i32, 2, 3, 4, 5];
    let indices = [1i32, 3, 1];
    let src = [10i32, 20, 30];
    let mut output = vec![0i32; 5];

    let input_shape = vec![5];
    let input_strides = vec![1];
    let src_shape = vec![3];
    let src_strides = vec![1];
    let indices_strides = vec![1];
    let input_offset = 0;
    let src_offset = 0;
    let indices_offset = 0;
    let dim = 0;

    call_scatter(
        scatter_add::I32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &input_strides,
        input_offset,
        indices.as_ptr(),
        &indices_strides,
        indices_offset,
        src.as_ptr() as *const std::ffi::c_void,
        &src_shape,
        &src_strides,
        src_offset,
        dim,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![1, 42, 3, 24, 5]); // 2+10+30=42, 4+20=24
}

#[test]
fn test_scatter_max_f32_1d() {
    // Input: [1, 2, 3, 4, 5]
    // Indices: [0, 2, 0]
    // Src: [10, 20, 5]
    // Output: [10, 2, 20, 4, 5]  (max(1,10,5)=10, max(3,20)=20)
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let indices = [0i32, 2, 0];
    let src = [10.0f32, 20.0, 5.0];
    let mut output = vec![0.0f32; 5];

    let input_shape = vec![5];
    let input_strides = vec![1];
    let src_shape = vec![3];
    let src_strides = vec![1];
    let indices_strides = vec![1];
    let input_offset = 0;
    let src_offset = 0;
    let indices_offset = 0;
    let dim = 0;

    call_scatter(
        scatter_max::F32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &input_strides,
        input_offset,
        indices.as_ptr(),
        &indices_strides,
        indices_offset,
        src.as_ptr() as *const std::ffi::c_void,
        &src_shape,
        &src_strides,
        src_offset,
        dim,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![10.0, 2.0, 20.0, 4.0, 5.0]);
}

#[test]
fn test_scatter_max_i32() {
    // Test with integer type
    let input = [5i32, 10, 15, 20, 25];
    let indices = [1i32, 3, 1];
    let src = [50i32, 30, 8];
    let mut output = vec![0i32; 5];

    let input_shape = vec![5];
    let input_strides = vec![1];
    let src_shape = vec![3];
    let src_strides = vec![1];
    let indices_strides = vec![1];
    let input_offset = 0;
    let src_offset = 0;
    let indices_offset = 0;
    let dim = 0;

    call_scatter(
        scatter_max::I32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &input_strides,
        input_offset,
        indices.as_ptr(),
        &indices_strides,
        indices_offset,
        src.as_ptr() as *const std::ffi::c_void,
        &src_shape,
        &src_strides,
        src_offset,
        dim,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![5, 50, 15, 30, 25]); // max(10,50,8)=50, max(20,30)=30
}

#[test]
fn test_scatter_min_f32_1d() {
    // Input: [10, 20, 30, 40, 50]
    // Indices: [0, 2, 0]
    // Src: [5, 25, 15]
    // Output: [5, 20, 25, 40, 50]  (min(10,5,15)=5, min(30,25)=25)
    let input = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    let indices = [0i32, 2, 0];
    let src = [5.0f32, 25.0, 15.0];
    let mut output = vec![0.0f32; 5];

    let input_shape = vec![5];
    let input_strides = vec![1];
    let src_shape = vec![3];
    let src_strides = vec![1];
    let indices_strides = vec![1];
    let input_offset = 0;
    let src_offset = 0;
    let indices_offset = 0;
    let dim = 0;

    call_scatter(
        scatter_min::F32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &input_strides,
        input_offset,
        indices.as_ptr(),
        &indices_strides,
        indices_offset,
        src.as_ptr() as *const std::ffi::c_void,
        &src_shape,
        &src_strides,
        src_offset,
        dim,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![5.0, 20.0, 25.0, 40.0, 50.0]);
}

#[test]
fn test_scatter_min_i32() {
    // Test with integer type
    let input = [50i32, 40, 30, 20, 10];
    let indices = [1i32, 3, 1];
    let src = [5i32, 15, 60];
    let mut output = vec![0i32; 5];

    let input_shape = vec![5];
    let input_strides = vec![1];
    let src_shape = vec![3];
    let src_strides = vec![1];
    let indices_strides = vec![1];
    let input_offset = 0;
    let src_offset = 0;
    let indices_offset = 0;
    let dim = 0;

    call_scatter(
        scatter_min::I32,
        &input_shape,
        input.as_ptr() as *const std::ffi::c_void,
        &input_strides,
        input_offset,
        indices.as_ptr(),
        &indices_strides,
        indices_offset,
        src.as_ptr() as *const std::ffi::c_void,
        &src_shape,
        &src_strides,
        src_offset,
        dim,
        output.as_mut_ptr() as *mut std::ffi::c_void,
    )
    .unwrap();

    assert_eq!(output, vec![50, 5, 30, 15, 10]); // min(40,5,60)=5, min(20,15)=15
}
