use half::{bf16, f16};
use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{
        call_reduce_window, reduce_window_max, reduce_window_mean, reduce_window_min, reduce_window_sum, Kernel,
    },
    metal::{create_command_buffer, Buffer, Device},
    utils::BufferOffset,
    RESOURCE_OPTIONS,
};
use std::ffi::c_void;

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

fn new_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let options = RESOURCE_OPTIONS;
    let ptr = data.as_ptr() as *const c_void;
    let size = std::mem::size_of_val(data);
    device.new_buffer_with_data(ptr, size, options).unwrap()
}

fn device() -> Device {
    Device::system_default().unwrap()
}

#[allow(clippy::too_many_arguments)]
fn run_reduce_window<T: Clone>(
    input: &[T],
    input_shape: &[usize],
    window_shape: &[usize],
    strides: &[usize],
    padding: &[usize],
    output_shape: &[usize],
    name: Kernel,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;
    let input_buffer = new_buffer(&device, input);

    let output_size: usize = output_shape.iter().product();
    let output = device
        .new_buffer(output_size * std::mem::size_of::<T>(), options)
        .unwrap();

    // Calculate strides for input
    let mut input_strides = vec![1; input_shape.len()];
    for i in (0..input_shape.len() - 1).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    call_reduce_window(
        &device,
        &command_buffer,
        &kernels,
        name,
        input_shape,
        BufferOffset::zero_offset(&input_buffer),
        &input_strides,
        0,
        window_shape,
        strides,
        padding,
        output_shape,
        &output,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, output_size)
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
fn test_reduce_window_max_bf16() {
    let input: Vec<bf16> = vec![1.0, 2.0, 3.0, 4.0].into_iter().map(bf16::from_f32).collect();
    let input_shape = vec![4];
    let window_shape = vec![2];
    let strides = vec![1];
    let padding = vec![0, 0];
    let output_shape = vec![3];

    let result: Vec<bf16> = run_reduce_window(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_max::BF16,
    );

    let expected: Vec<bf16> = vec![2.0, 3.0, 4.0].into_iter().map(bf16::from_f32).collect();
    assert_eq!(result, expected);
}

#[test]
fn test_reduce_window_max_f16() {
    let input: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0].into_iter().map(f16::from_f32).collect();
    let input_shape = vec![4];
    let window_shape = vec![2];
    let strides = vec![1];
    let padding = vec![0, 0];
    let output_shape = vec![3];

    let result: Vec<f16> = run_reduce_window(
        &input,
        &input_shape,
        &window_shape,
        &strides,
        &padding,
        &output_shape,
        reduce_window_max::F16,
    );

    let expected: Vec<f16> = vec![2.0, 3.0, 4.0].into_iter().map(f16::from_f32).collect();
    assert_eq!(result, expected);
}
