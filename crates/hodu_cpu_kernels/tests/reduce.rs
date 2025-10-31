use hodu_cpu_kernels::*;

fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t * b) / b).collect()
}

// Helper function to calculate strides from shape
fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// Helper function to calculate output shape
fn calculate_output_shape(shape: &[usize], reduce_dims: &[usize], keep_dim: bool) -> Vec<usize> {
    let mut output_shape = shape.to_vec();
    for &dim in reduce_dims.iter() {
        if keep_dim {
            output_shape[dim] = 1;
        } else {
            output_shape[dim] = 0;
        }
    }
    if !keep_dim {
        output_shape.retain(|&size| size != 0);
        if output_shape.is_empty() {
            output_shape = vec![1];
        }
    }
    output_shape
}

// reduce - sum
#[test]
fn test_reduce_sum_f32() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3]; // 2x3 matrix
    let reduce_dims = vec![1]; // reduce along columns
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0.0f32; output_size];

    // Metadata: [shape..., strides..., offset, output_shape_len, output_shape...,
    //            num_reduce_dims, reduce_dims..., keep_dim, reduce_size]
    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0); // offset
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_sum::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[1, 2, 3], [4, 5, 6]] -> sum along dim 1 -> [6, 15]
    assert_eq!(approx(output, 4), vec![6.0, 15.0]);
}

#[test]
fn test_reduce_sum_f32_dim0() {
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let reduce_dims = vec![0]; // reduce along rows
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0.0f32; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_sum::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[1, 2], [3, 4]] -> sum along dim 0 -> [4, 6]
    assert_eq!(approx(output, 4), vec![4.0, 6.0]);
}

#[test]
fn test_reduce_sum_3d() {
    let input: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let shape = vec![2, 3, 4]; // 2x3x4 tensor
    let reduce_dims = vec![2]; // reduce along last dimension
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0.0f32; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_sum::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Sum along last dimension (groups of 4)
    // [1,2,3,4] -> 10, [5,6,7,8] -> 26, [9,10,11,12] -> 42,
    // [13,14,15,16] -> 58, [17,18,19,20] -> 74, [21,22,23,24] -> 90
    assert_eq!(approx(output, 4), vec![10.0, 26.0, 42.0, 58.0, 74.0, 90.0]);
}

#[test]
fn test_reduce_sum_f32_keep_dim() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3]; // 2x3 matrix
    let reduce_dims = vec![1]; // reduce along columns
    let keep_dim = true;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0.0f32; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_sum::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[1, 2, 3], [4, 5, 6]] -> sum along dim 1 with keep_dim -> [[6], [15]]
    // Output shape: [2, 1], flattened: [6, 15]
    assert_eq!(approx(output, 4), vec![6.0, 15.0]);
}

#[test]
fn test_reduce_sum_i32() {
    let input = [1i32, 2, 3, 4, 5, 6];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0i32; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_sum::I32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[1, 2, 3], [4, 5, 6]] -> sum along dim 1 -> [6, 15]
    assert_eq!(output, vec![6, 15]);
}

// reduce - mean
#[test]
fn test_reduce_mean_f32() {
    let input = [2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0.0f32; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_mean::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[2, 4, 6], [8, 10, 12]] -> mean along dim 1 -> [4, 10]
    assert_eq!(approx(output, 4), vec![4.0, 10.0]);
}

// reduce - max
#[test]
fn test_reduce_max_f32() {
    let input = [1.0f32, 5.0, 3.0, 2.0, 8.0, 1.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0.0f32; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_max::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[1, 5, 3], [2, 8, 1]] -> max along dim 1 -> [5, 8]
    assert_eq!(approx(output, 4), vec![5.0, 8.0]);
}

#[test]
fn test_reduce_max_f32_keep_dim() {
    let input = [1.0f32, 5.0, 3.0, 2.0, 8.0, 1.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];
    let keep_dim = true;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0.0f32; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_max::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[1, 5, 3], [2, 8, 1]] -> max along dim 1 with keep_dim -> [[5], [8]]
    assert_eq!(approx(output, 4), vec![5.0, 8.0]);
}

// reduce - min
#[test]
fn test_reduce_min_f32() {
    let input = [1.0f32, 5.0, 3.0, 2.0, 8.0, 1.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0.0f32; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_min::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[1, 5, 3], [2, 8, 1]] -> min along dim 1 -> [1, 1]
    assert_eq!(approx(output, 4), vec![1.0, 1.0]);
}

// reduce - prod
#[test]
fn test_reduce_prod_f32() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0.0f32; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_prod::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[1, 2, 3], [4, 5, 6]] -> prod along dim 1 -> [6, 120]
    assert_eq!(approx(output, 4), vec![6.0, 120.0]);
}

// reduce - std
#[test]
fn test_reduce_std_f32() {
    // Test data: [1, 2, 3, 4, 5, 6]
    // Shape: [2, 3] -> [[1, 2, 3], [4, 5, 6]]
    // Reduce along dim 1:
    // Row 0: [1, 2, 3] -> mean=2, var=((1-2)^2+(2-2)^2+(3-2)^2)/3 = (1+0+1)/3 = 0.6667, std=0.8165
    // Row 1: [4, 5, 6] -> mean=5, var=((4-5)^2+(5-5)^2+(6-5)^2)/3 = (1+0+1)/3 = 0.6667, std=0.8165
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0.0f32; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_std::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Expected std ≈ 0.8165 for both rows
    assert_eq!(approx(output, 2), vec![0.82, 0.82]);
}

// reduce - var
#[test]
fn test_reduce_var_f32() {
    // Test data: [1, 2, 3, 4, 5, 6]
    // Shape: [2, 3] -> [[1, 2, 3], [4, 5, 6]]
    // Reduce along dim 1:
    // Row 0: [1, 2, 3] -> mean=2, var=((1-2)^2+(2-2)^2+(3-2)^2)/3 = 0.6667
    // Row 1: [4, 5, 6] -> mean=5, var=((4-5)^2+(5-5)^2+(6-5)^2)/3 = 0.6667
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0.0f32; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_var::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Expected var ≈ 0.6667 for both rows
    assert_eq!(approx(output, 2), vec![0.67, 0.67]);
}

// reduce - norm
#[test]
fn test_reduce_norm_f32() {
    let input = [3.0f32, 4.0, 5.0, 12.0];
    let shape = vec![2, 2];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0.0f32; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_norm::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[3, 4], [5, 12]] -> L2 norm along dim 1 -> [5, 13]
    // sqrt(3^2 + 4^2) = sqrt(25) = 5
    // sqrt(5^2 + 12^2) = sqrt(169) = 13
    assert_eq!(approx(output, 4), vec![5.0, 13.0]);
}

// reduce - argmax
#[test]
fn test_reduce_argmax_f32() {
    let input = [1.0f32, 5.0, 3.0, 2.0, 8.0, 1.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0i32; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_argmax::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[1, 5, 3], [2, 8, 1]] -> argmax along dim 1 -> [1, 1]
    assert_eq!(output, vec![1, 1]);
}

// reduce - argmin
#[test]
fn test_reduce_argmin_f32() {
    let input = [1.0f32, 5.0, 3.0, 2.0, 8.0, 1.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0i32; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_argmin::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[1, 5, 3], [2, 8, 1]] -> argmin along dim 1 -> [0, 2]
    assert_eq!(output, vec![0, 2]);
}

// reduce - any
#[test]
fn test_reduce_any_f32() {
    let input = [0.0f32, 1.0, 0.0, 0.0, 2.0, 0.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0u8; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_any::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[0, 1, 0], [0, 2, 0]] -> any along dim 1 -> [true, true]
    assert_eq!(output, vec![1, 1]);
}

#[test]
fn test_reduce_any_f32_all_zeros() {
    let input = [0.0f32, 0.0, 0.0, 0.0];
    let shape = vec![2, 2];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0u8; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_any::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[0, 0], [0, 0]] -> any along dim 1 -> [false, false]
    assert_eq!(output, vec![0, 0]);
}

#[test]
fn test_reduce_any_i32() {
    let input = [0i32, 1, 0, 0, 2, 0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0u8; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_any::I32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[0, 1, 0], [0, 2, 0]] -> any along dim 1 -> [true, true]
    assert_eq!(output, vec![1, 1]);
}

// reduce - all
#[test]
fn test_reduce_all_f32() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0u8; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_all::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[1, 2, 3], [4, 5, 6]] -> all along dim 1 -> [true, true]
    assert_eq!(output, vec![1, 1]);
}

#[test]
fn test_reduce_all_f32_with_zero() {
    let input = [1.0f32, 0.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0u8; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_all::F32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[1, 0, 3], [4, 5, 6]] -> all along dim 1 -> [false, true]
    assert_eq!(output, vec![0, 1]);
}

#[test]
fn test_reduce_all_i32() {
    let input = [1i32, 2, 3, 4, 5, 6];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];
    let keep_dim = false;
    let strides = calculate_strides(&shape);
    let output_shape = calculate_output_shape(&shape, &reduce_dims, keep_dim);
    let output_size: usize = output_shape.iter().product();
    let reduce_size: usize = reduce_dims.iter().map(|&d| shape[d]).product();
    let mut output = vec![0u8; output_size];

    let mut metadata = vec![shape.len()];
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape.len());
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_reduce(
        reduce_all::I32,
        input.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // [[1, 2, 3], [4, 5, 6]] -> all along dim 1 -> [true, true]
    assert_eq!(output, vec![1, 1]);
}
