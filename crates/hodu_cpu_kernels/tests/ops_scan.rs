use hodu_cpu_kernels::*;

// Helper function to calculate strides from shape
fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// Helper function to build cumsum metadata
// Layout: [num_els, num_dims, shape..., strides..., offset, dim]
fn build_cumsum_metadata(shape: &[usize], strides: &[usize], offset: usize, dim: usize) -> Vec<usize> {
    let num_els: usize = shape.iter().product();
    let num_dims = shape.len();
    let mut metadata = vec![num_els, num_dims];
    metadata.extend(shape);
    metadata.extend(strides);
    metadata.push(offset);
    metadata.push(dim);
    metadata
}

// cumsum - 1D tensor
#[test]
fn test_cumsum_1d_f32() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let shape = vec![5];
    let strides = calculate_strides(&shape);
    let mut output = vec![0.0f32; 5];

    let metadata = build_cumsum_metadata(&shape, &strides, 0, 0);

    call_ops_cumsum(
        cumsum::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [1, 2, 3, 4, 5] -> cumsum -> [1, 3, 6, 10, 15]
    assert_eq!(output, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
}

// cumsum - 2D tensor along dim 0
#[test]
fn test_cumsum_2d_dim0_f32() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3]; // [[1, 2, 3], [4, 5, 6]]
    let strides = calculate_strides(&shape);
    let mut output = vec![0.0f32; 6];

    let metadata = build_cumsum_metadata(&shape, &strides, 0, 0);

    call_ops_cumsum(
        cumsum::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // cumsum along dim 0:
    // [[1, 2, 3], [4, 5, 6]] -> [[1, 2, 3], [1+4, 2+5, 3+6]] = [[1, 2, 3], [5, 7, 9]]
    assert_eq!(output, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
}

// cumsum - 2D tensor along dim 1
#[test]
fn test_cumsum_2d_dim1_f32() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3]; // [[1, 2, 3], [4, 5, 6]]
    let strides = calculate_strides(&shape);
    let mut output = vec![0.0f32; 6];

    let metadata = build_cumsum_metadata(&shape, &strides, 0, 1);

    call_ops_cumsum(
        cumsum::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // cumsum along dim 1:
    // [[1, 2, 3], [4, 5, 6]] -> [[1, 1+2, 1+2+3], [4, 4+5, 4+5+6]] = [[1, 3, 6], [4, 9, 15]]
    assert_eq!(output, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
}

// cumsum - 3D tensor
#[test]
fn test_cumsum_3d_f32() {
    // Shape [2, 2, 3]
    let input: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let shape = vec![2, 2, 3];
    let strides = calculate_strides(&shape);
    let mut output = vec![0.0f32; 12];

    // cumsum along dim 2 (last dimension)
    let metadata = build_cumsum_metadata(&shape, &strides, 0, 2);

    call_ops_cumsum(
        cumsum::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Input: [[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]
    // cumsum along dim 2:
    // [[[1,3,6], [4,9,15]], [[7,15,24], [10,21,33]]]
    assert_eq!(
        output,
        vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0, 7.0, 15.0, 24.0, 10.0, 21.0, 33.0]
    );
}

// cumsum - 3D tensor along middle dimension
#[test]
fn test_cumsum_3d_dim1_f32() {
    // Shape [2, 2, 3]
    let input: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let shape = vec![2, 2, 3];
    let strides = calculate_strides(&shape);
    let mut output = vec![0.0f32; 12];

    // cumsum along dim 1
    let metadata = build_cumsum_metadata(&shape, &strides, 0, 1);

    call_ops_cumsum(
        cumsum::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Input: [[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]
    // cumsum along dim 1:
    // [[[1,2,3], [1+4,2+5,3+6]], [[7,8,9], [7+10,8+11,9+12]]]
    // = [[[1,2,3], [5,7,9]], [[7,8,9], [17,19,21]]]
    assert_eq!(
        output,
        vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 7.0, 8.0, 9.0, 17.0, 19.0, 21.0]
    );
}

// cumsum - integer types
#[test]
fn test_cumsum_i32() {
    let input = [1i32, 2, 3, 4, 5, 6];
    let shape = vec![2, 3];
    let strides = calculate_strides(&shape);
    let mut output = vec![0i32; 6];

    let metadata = build_cumsum_metadata(&shape, &strides, 0, 1);

    call_ops_cumsum(
        cumsum::I32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // cumsum along dim 1
    assert_eq!(output, vec![1, 3, 6, 4, 9, 15]);
}

#[test]
fn test_cumsum_u32() {
    let input = [1u32, 2, 3, 4, 5, 6];
    let shape = vec![2, 3];
    let strides = calculate_strides(&shape);
    let mut output = vec![0u32; 6];

    let metadata = build_cumsum_metadata(&shape, &strides, 0, 1);

    call_ops_cumsum(
        cumsum::U32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // cumsum along dim 1
    assert_eq!(output, vec![1, 3, 6, 4, 9, 15]);
}

// cumsum - with offset
#[test]
fn test_cumsum_with_offset_f32() {
    let input = [0.0f32, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]; // offset 2 -> [1, 2, 3, 4, 5]
    let shape = vec![5];
    let strides = calculate_strides(&shape);
    let mut output = vec![0.0f32; 5];

    let metadata = build_cumsum_metadata(&shape, &strides, 2, 0);

    call_ops_cumsum(
        cumsum::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [1, 2, 3, 4, 5] -> cumsum -> [1, 3, 6, 10, 15]
    assert_eq!(output, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
}

// cumsum - scalar (0-dim tensor)
#[test]
fn test_cumsum_scalar_f32() {
    let input = [42.0f32];
    let mut output = vec![0.0f32; 1];

    // For scalar, num_dims = 0
    // metadata: [num_els, num_dims, offset, dim]
    let metadata = vec![1, 0, 0, 0];

    call_ops_cumsum(
        cumsum::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Scalar cumsum is just the value itself
    assert_eq!(output, vec![42.0]);
}

// cumsum - non-contiguous stride
#[test]
fn test_cumsum_non_contiguous_f32() {
    // Original: [[1, 2, 3], [4, 5, 6]] with shape [2, 3], strides [3, 1]
    // Transposed view: shape [3, 2], strides [1, 3]
    // Logical: [[1, 4], [2, 5], [3, 6]]
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![3, 2]; // transposed shape
    let strides = vec![1, 3]; // transposed strides
    let mut output = vec![0.0f32; 6];

    // cumsum along dim 1
    let metadata = build_cumsum_metadata(&shape, &strides, 0, 1);

    call_ops_cumsum(
        cumsum::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Logical input: [[1, 4], [2, 5], [3, 6]]
    // cumsum along dim 1: [[1, 1+4], [2, 2+5], [3, 3+6]] = [[1, 5], [2, 7], [3, 9]]
    // Output in row-major: [1, 5, 2, 7, 3, 9]
    assert_eq!(output, vec![1.0, 5.0, 2.0, 7.0, 3.0, 9.0]);
}

// cumsum - f64
#[test]
fn test_cumsum_f64() {
    let input = [1.0f64, 2.0, 3.0, 4.0, 5.0];
    let shape = vec![5];
    let strides = calculate_strides(&shape);
    let mut output = vec![0.0f64; 5];

    let metadata = build_cumsum_metadata(&shape, &strides, 0, 0);

    call_ops_cumsum(
        cumsum::F64,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
}

// cumsum - i64
#[test]
fn test_cumsum_i64() {
    let input = [1i64, 2, 3, 4, 5];
    let shape = vec![5];
    let strides = calculate_strides(&shape);
    let mut output = vec![0i64; 5];

    let metadata = build_cumsum_metadata(&shape, &strides, 0, 0);

    call_ops_cumsum(
        cumsum::I64,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![1, 3, 6, 10, 15]);
}

// cumsum - u8
#[test]
fn test_cumsum_u8() {
    let input = [1u8, 2, 3, 4, 5];
    let shape = vec![5];
    let strides = calculate_strides(&shape);
    let mut output = vec![0u8; 5];

    let metadata = build_cumsum_metadata(&shape, &strides, 0, 0);

    call_ops_cumsum(
        cumsum::U8,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output, vec![1, 3, 6, 10, 15]);
}

// cumsum - i8
#[test]
fn test_cumsum_i8() {
    let input = [1i8, 2, 3, -4, 5];
    let shape = vec![5];
    let strides = calculate_strides(&shape);
    let mut output = vec![0i8; 5];

    let metadata = build_cumsum_metadata(&shape, &strides, 0, 0);

    call_ops_cumsum(
        cumsum::I8,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [1, 2, 3, -4, 5] -> [1, 3, 6, 2, 7]
    assert_eq!(output, vec![1, 3, 6, 2, 7]);
}
