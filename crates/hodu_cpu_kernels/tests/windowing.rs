use hodu_cpu_kernels::*;
use std::ffi::c_void;

fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[allow(clippy::too_many_arguments)]
fn run_reduce_window_f32(
    input: &[f32],
    input_shape: &[usize],
    window_shape: &[usize],
    strides: &[usize],
    padding: &[usize],
    output_shape: &[usize],
    name: Kernel,
) -> Vec<f32> {
    let output_size: usize = output_shape.iter().product();
    let mut output = vec![0.0f32; output_size];

    let input_strides = calculate_strides(input_shape);

    let mut metadata = Vec::new();
    metadata.extend(input_shape);
    metadata.extend(&input_strides);
    metadata.push(0);
    metadata.extend(window_shape);
    metadata.extend(strides);
    metadata.extend(padding);
    metadata.extend(output_shape);

    call_reduce_window(
        name,
        input.as_ptr() as *const c_void,
        output.as_mut_ptr() as *mut c_void,
        output_size,
        input_shape.len(),
        &metadata,
    )
    .unwrap();

    output
}

#[test]
fn test_reduce_window_max_1d() {
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let input_shape = vec![5];
    let window_shape = vec![2];
    let strides = vec![1];
    let padding = vec![0, 0];
    let output_shape = vec![4];

    let result = run_reduce_window_f32(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_max::F32,
    );

    assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_reduce_window_sum_1d() {
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input_shape = vec![4];
    let window_shape = vec![2];
    let strides = vec![2];
    let padding = vec![0, 0];
    let output_shape = vec![2];

    let result = run_reduce_window_f32(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_sum::F32,
    );

    assert_eq!(result, vec![3.0, 7.0]);
}

#[test]
fn test_reduce_window_mean_1d() {
    let input: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0];
    let input_shape = vec![4];
    let window_shape = vec![2];
    let strides = vec![2];
    let padding = vec![0, 0];
    let output_shape = vec![2];

    let result = run_reduce_window_f32(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_mean::F32,
    );

    assert_eq!(result, vec![3.0, 7.0]);
}

#[test]
fn test_reduce_window_min_1d() {
    let input: Vec<f32> = vec![5.0, 2.0, 8.0, 1.0];
    let input_shape = vec![4];
    let window_shape = vec![2];
    let strides = vec![1];
    let padding = vec![0, 0];
    let output_shape = vec![3];

    let result = run_reduce_window_f32(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_min::F32,
    );

    assert_eq!(result, vec![2.0, 2.0, 1.0]);
}

#[test]
fn test_reduce_window_max_2d() {
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let input_shape = vec![4, 4];
    let window_shape = vec![2, 2];
    let strides = vec![2, 2];
    let padding = vec![0, 0, 0, 0];
    let output_shape = vec![2, 2];

    let result = run_reduce_window_f32(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_max::F32,
    );

    assert_eq!(result, vec![6.0, 8.0, 14.0, 16.0]);
}

#[test]
fn test_reduce_window_sum_2d() {
    let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let input_shape = vec![3, 3];
    let window_shape = vec![2, 2];
    let strides = vec![1, 1];
    let padding = vec![0, 0, 0, 0];
    let output_shape = vec![2, 2];

    let result = run_reduce_window_f32(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_sum::F32,
    );

    assert_eq!(result, vec![12.0, 16.0, 24.0, 28.0]);
}

#[test]
fn test_reduce_window_mean_2d() {
    let input: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0];
    let input_shape = vec![2, 2];
    let window_shape = vec![2, 2];
    let strides = vec![1, 1];
    let padding = vec![0, 0, 0, 0];
    let output_shape = vec![1, 1];

    let result = run_reduce_window_f32(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_mean::F32,
    );

    assert_eq!(result, vec![5.0]);
}

#[test]
fn test_reduce_window_min_2d() {
    let input: Vec<f32> = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let input_shape = vec![3, 3];
    let window_shape = vec![2, 2];
    let strides = vec![1, 1];
    let padding = vec![0, 0, 0, 0];
    let output_shape = vec![2, 2];

    let result = run_reduce_window_f32(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_min::F32,
    );

    assert_eq!(result, vec![5.0, 4.0, 2.0, 1.0]);
}

#[test]
fn test_reduce_window_max_3d() {
    let input: Vec<f32> = (1..=27).map(|x| x as f32).collect();
    let input_shape = vec![3, 3, 3];
    let window_shape = vec![2, 2, 2];
    let strides = vec![1, 1, 1];
    let padding = vec![0, 0, 0, 0, 0, 0];
    let output_shape = vec![2, 2, 2];

    let result = run_reduce_window_f32(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_max::F32,
    );

    assert_eq!(result, vec![14.0, 15.0, 17.0, 18.0, 23.0, 24.0, 26.0, 27.0]);
}

#[test]
fn test_reduce_window_sum_stride_2() {
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let input_shape = vec![8];
    let window_shape = vec![3];
    let strides = vec![2];
    let padding = vec![0, 0];
    let output_shape = vec![3];

    let result = run_reduce_window_f32(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_sum::F32,
    );

    assert_eq!(result, vec![6.0, 12.0, 18.0]);
}

#[test]
fn test_reduce_window_mean_stride_2() {
    let input: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let input_shape = vec![6];
    let window_shape = vec![2];
    let strides = vec![2];
    let padding = vec![0, 0];
    let output_shape = vec![3];

    let result = run_reduce_window_f32(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_mean::F32,
    );

    assert_eq!(result, vec![1.5, 3.5, 5.5]);
}

#[test]
fn test_reduce_window_max_with_padding() {
    let input: Vec<f32> = vec![1.0, 2.0, 3.0];
    let input_shape = vec![3];
    let window_shape = vec![3];
    let strides = vec![1];
    let padding = vec![1, 1];
    let output_shape = vec![3];

    let result = run_reduce_window_f32(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_max::F32,
    );

    assert_eq!(result, vec![2.0, 3.0, 3.0]);
}

#[test]
fn test_reduce_window_sum_with_padding() {
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input_shape = vec![4];
    let window_shape = vec![2];
    let strides = vec![1];
    let padding = vec![1, 1];
    let output_shape = vec![6];

    let result = run_reduce_window_f32(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_sum::F32,
    );

    assert_eq!(result, vec![1.0, 3.0, 5.0, 7.0, 4.0, 0.0]);
}
