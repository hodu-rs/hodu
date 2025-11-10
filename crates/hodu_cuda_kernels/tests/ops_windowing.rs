use hodu_cuda_kernels::{compat::*, kernel::Kernels, kernels::*};

fn device() -> Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

fn kernels() -> Kernels {
    Kernels::new()
}

#[allow(clippy::too_many_arguments)]
fn run_reduce_window<T: cudarc::driver::DeviceRepr + Clone>(
    input: &[T],
    input_shape: &[usize],
    window_shape: &[usize],
    strides: &[usize],
    padding: &[usize],
    output_shape: &[usize],
    kernel: hodu_cuda_kernels::kernels::Kernel,
) -> Vec<T> {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let output_size: usize = output_shape.iter().product();
    let mut output: cudarc::driver::CudaSlice<T> = unsafe { stream.alloc(output_size).unwrap() };

    // Calculate strides for input
    let mut input_strides = vec![1; input_shape.len()];
    for i in (0..input_shape.len() - 1).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // Build metadata: [output_size, num_dims, input_shape..., input_strides..., input_offset, window_shape..., strides..., padding..., output_shape...]
    let num_dims = input_shape.len();
    let mut metadata = Vec::new();
    metadata.push(output_size);
    metadata.push(num_dims);
    metadata.extend_from_slice(input_shape);
    metadata.extend_from_slice(&input_strides);
    metadata.push(0); // input_offset
    metadata.extend_from_slice(window_shape);
    metadata.extend_from_slice(strides);
    metadata.extend_from_slice(padding);
    metadata.extend_from_slice(output_shape);

    call_ops_reduce_window(kernel, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![unsafe { core::mem::zeroed() }; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    results
}

#[test]
fn test_reduce_window_max_1d() {
    // Input: [1, 2, 3, 4, 5]
    // Window: [2], Stride: [1], Padding: [0, 0]
    // Output: [2, 3, 4, 5] (max of sliding windows)
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let input_shape = vec![5];
    let window_shape = vec![2];
    let strides = vec![1];
    let padding = vec![0, 0];
    let output_shape = vec![4];

    let result: Vec<f32> = run_reduce_window(
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
    // Input: [1, 2, 3, 4]
    // Window: [2], Stride: [2], Padding: [0, 0]
    // Output: [3, 7] (sum of non-overlapping windows)
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input_shape = vec![4];
    let window_shape = vec![2];
    let strides = vec![2];
    let padding = vec![0, 0];
    let output_shape = vec![2];

    let result: Vec<f32> = run_reduce_window(
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
    // Input: [2, 4, 6, 8]
    // Window: [2], Stride: [2], Padding: [0, 0]
    // Output: [3, 7] (mean of non-overlapping windows)
    let input: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0];
    let input_shape = vec![4];
    let window_shape = vec![2];
    let strides = vec![2];
    let padding = vec![0, 0];
    let output_shape = vec![2];

    let result: Vec<f32> = run_reduce_window(
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
fn test_reduce_window_max_2d() {
    // Input: 4x4 matrix
    // [[1,  2,  3,  4],
    //  [5,  6,  7,  8],
    //  [9,  10, 11, 12],
    //  [13, 14, 15, 16]]
    // Window: [2, 2], Stride: [2, 2], Padding: [0, 0, 0, 0]
    // Output: [[6, 8], [14, 16]] (max pooling 2x2)
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let input_shape = vec![4, 4];
    let window_shape = vec![2, 2];
    let strides = vec![2, 2];
    let padding = vec![0, 0, 0, 0]; // [pad_before_h, pad_after_h, pad_before_w, pad_after_w]
    let output_shape = vec![2, 2];

    let result: Vec<f32> = run_reduce_window(
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
    // Input: 3x3 matrix
    // [[1, 2, 3],
    //  [4, 5, 6],
    //  [7, 8, 9]]
    // Window: [2, 2], Stride: [1, 1], Padding: [0, 0, 0, 0]
    // Output: [[12, 16], [24, 28]] (sum of sliding 2x2 windows)
    let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let input_shape = vec![3, 3];
    let window_shape = vec![2, 2];
    let strides = vec![1, 1];
    let padding = vec![0, 0, 0, 0];
    let output_shape = vec![2, 2];

    let result: Vec<f32> = run_reduce_window(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_sum::F32,
    );

    // Windows: [[1,2,4,5], [2,3,5,6], [4,5,7,8], [5,6,8,9]]
    assert_eq!(result, vec![12.0, 16.0, 24.0, 28.0]);
}

#[test]
fn test_reduce_window_min_2d() {
    // Input: 3x3 matrix
    // [[9, 8, 7],
    //  [6, 5, 4],
    //  [3, 2, 1]]
    // Window: [2, 2], Stride: [1, 1], Padding: [0, 0, 0, 0]
    // Output: [[5, 4], [2, 1]] (min of sliding 2x2 windows)
    let input: Vec<f32> = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let input_shape = vec![3, 3];
    let window_shape = vec![2, 2];
    let strides = vec![1, 1];
    let padding = vec![0, 0, 0, 0];
    let output_shape = vec![2, 2];

    let result: Vec<f32> = run_reduce_window(
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
fn test_reduce_window_max_1d_stride3() {
    // Input: [1, 2, 3, 4, 5, 6, 7, 8, 9]
    // Window: [3], Stride: [3], Padding: [0, 0]
    // Output: [3, 6, 9]
    let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let input_shape = vec![9];
    let window_shape = vec![3];
    let strides = vec![3];
    let padding = vec![0, 0];
    let output_shape = vec![3];

    let result: Vec<f32> = run_reduce_window(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_max::F32,
    );

    assert_eq!(result, vec![3.0, 6.0, 9.0]);
}

#[test]
fn test_reduce_window_sum_2d_3x3() {
    // Input: 5x5 matrix, Window: [3, 3], Stride: [2, 2]
    let input: Vec<f32> = (1..=25).map(|x| x as f32).collect();
    let input_shape = vec![5, 5];
    let window_shape = vec![3, 3];
    let strides = vec![2, 2];
    let padding = vec![0, 0, 0, 0];
    let output_shape = vec![2, 2];

    let result: Vec<f32> = run_reduce_window(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_sum::F32,
    );

    // First window: elements [1,2,3,6,7,8,11,12,13] = 1+2+3+6+7+8+11+12+13 = 63
    // Second window: elements [3,4,5,8,9,10,13,14,15] = 3+4+5+8+9+10+13+14+15 = 81
    // Third window: elements [11,12,13,16,17,18,21,22,23] = 11+12+13+16+17+18+21+22+23 = 153
    // Fourth window: elements [13,14,15,18,19,20,23,24,25] = 13+14+15+18+19+20+23+24+25 = 171
    assert_eq!(result, vec![63.0, 81.0, 153.0, 171.0]);
}
