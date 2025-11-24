#![allow(clippy::identity_op)]

use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{
        call_ops_conv, call_ops_conv_grad_weight, conv1d, conv1d_grad_weight, conv2d, conv2d_grad_weight, conv3d,
        conv3d_grad_weight, conv_transpose1d, conv_transpose1d_grad_weight, conv_transpose2d,
        conv_transpose2d_grad_weight, conv_transpose3d, conv_transpose3d_grad_weight, Kernel,
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

fn new_buffer_zeroed<T>(device: &Device, count: usize) -> Buffer {
    let options = RESOURCE_OPTIONS;
    let size = count * std::mem::size_of::<T>();
    let buffer = device.new_buffer(size, options).unwrap();
    // Zero initialize
    unsafe {
        let ptr = buffer.contents();
        std::ptr::write_bytes(ptr, 0, size);
    }
    buffer
}

fn device() -> Device {
    Device::system_default().unwrap()
}

// CONV1D
#[allow(clippy::too_many_arguments)]
fn run_conv1d<T: Clone>(
    input: &[T],
    weight: &[T],
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    input_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    kernel: Kernel,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;

    let input_buffer = new_buffer(&device, input);
    let weight_buffer = new_buffer(&device, weight);

    let output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    let output_size = batch * out_channels * output_length;

    let output = device
        .new_buffer(output_size * std::mem::size_of::<T>(), options)
        .unwrap();

    let metadata = vec![
        output_size,
        batch,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        output_length,
        stride,
        padding,
        1,
        0,
        0,
    ];

    call_ops_conv(
        kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&weight_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, output_size)
}

// CONV2D
#[allow(clippy::too_many_arguments)]
fn run_conv2d<T: Clone>(
    input: &[T],
    weight: &[T],
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    input_height: usize,
    input_width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    kernel: Kernel,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;

    let input_buffer = new_buffer(&device, input);
    let weight_buffer = new_buffer(&device, weight);

    let output_height = (input_height + 2 * padding_h - kernel_h) / stride_h + 1;
    let output_width = (input_width + 2 * padding_w - kernel_w) / stride_w + 1;
    let output_size = batch * out_channels * output_height * output_width;

    let output = device
        .new_buffer(output_size * std::mem::size_of::<T>(), options)
        .unwrap();

    let metadata = vec![
        output_size,
        batch,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_h,
        kernel_w,
        output_height,
        output_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        1,
        1,
        0,
        0,
    ];

    call_ops_conv(
        kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&weight_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, output_size)
}

// CONV3D
#[allow(clippy::too_many_arguments)]
fn run_conv3d<T: Clone>(
    input: &[T],
    weight: &[T],
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    input_depth: usize,
    input_height: usize,
    input_width: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    padding_d: usize,
    padding_h: usize,
    padding_w: usize,
    kernel: Kernel,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;

    let input_buffer = new_buffer(&device, input);
    let weight_buffer = new_buffer(&device, weight);

    let output_depth = (input_depth + 2 * padding_d - kernel_d) / stride_d + 1;
    let output_height = (input_height + 2 * padding_h - kernel_h) / stride_h + 1;
    let output_width = (input_width + 2 * padding_w - kernel_w) / stride_w + 1;
    let output_size = batch * out_channels * output_depth * output_height * output_width;

    let output = device
        .new_buffer(output_size * std::mem::size_of::<T>(), options)
        .unwrap();

    let metadata = vec![
        output_size,
        batch,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_d,
        kernel_h,
        kernel_w,
        output_depth,
        output_height,
        output_width,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        1,
        1,
        1,
        0,
        0,
    ];

    call_ops_conv(
        kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&weight_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, output_size)
}

// CONV_TRANSPOSE1D
#[allow(clippy::too_many_arguments)]
fn run_conv_transpose1d<T: Clone>(
    input: &[T],
    weight: &[T],
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    input_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    kernel: Kernel,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;

    let input_buffer = new_buffer(&device, input);
    let weight_buffer = new_buffer(&device, weight);

    let output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    let output_size = batch * out_channels * output_length;

    let output = device
        .new_buffer(output_size * std::mem::size_of::<T>(), options)
        .unwrap();

    let metadata = vec![
        output_size,
        batch,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        output_length,
        stride,
        padding,
        1,
        output_padding,
        0,
        0,
    ];

    call_ops_conv(
        kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&weight_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, output_size)
}

// CONV_TRANSPOSE2D
#[allow(clippy::too_many_arguments)]
fn run_conv_transpose2d<T: Clone>(
    input: &[T],
    weight: &[T],
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    input_height: usize,
    input_width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    output_padding_h: usize,
    output_padding_w: usize,
    kernel: Kernel,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;

    let input_buffer = new_buffer(&device, input);
    let weight_buffer = new_buffer(&device, weight);

    let output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    let output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;
    let output_size = batch * out_channels * output_height * output_width;

    let output = device
        .new_buffer(output_size * std::mem::size_of::<T>(), options)
        .unwrap();

    let metadata = vec![
        output_size,
        batch,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_h,
        kernel_w,
        output_height,
        output_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        1,
        1,
        output_padding_h,
        output_padding_w,
        0,
        0,
    ];

    call_ops_conv(
        kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&weight_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, output_size)
}

// CONV_TRANSPOSE3D
#[allow(clippy::too_many_arguments)]
fn run_conv_transpose3d<T: Clone>(
    input: &[T],
    weight: &[T],
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    input_depth: usize,
    input_height: usize,
    input_width: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    padding_d: usize,
    padding_h: usize,
    padding_w: usize,
    output_padding_d: usize,
    output_padding_h: usize,
    output_padding_w: usize,
    kernel: Kernel,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;

    let input_buffer = new_buffer(&device, input);
    let weight_buffer = new_buffer(&device, weight);

    let output_depth = (input_depth - 1) * stride_d + kernel_d - 2 * padding_d + output_padding_d;
    let output_height = (input_height - 1) * stride_h + kernel_h - 2 * padding_h + output_padding_h;
    let output_width = (input_width - 1) * stride_w + kernel_w - 2 * padding_w + output_padding_w;
    let output_size = batch * out_channels * output_depth * output_height * output_width;

    let output = device
        .new_buffer(output_size * std::mem::size_of::<T>(), options)
        .unwrap();

    let metadata = vec![
        output_size,
        batch,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_d,
        kernel_h,
        kernel_w,
        output_depth,
        output_height,
        output_width,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        1,
        1,
        1,
        output_padding_d,
        output_padding_h,
        output_padding_w,
        0,
        0,
    ];

    call_ops_conv(
        kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&weight_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, output_size)
}

// ============================================================================
// TESTS
// ============================================================================

#[test]
fn test_conv1d_simple_f32() {
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight: Vec<f32> = vec![1.0, 0.0, -1.0];

    let result: Vec<f32> = run_conv1d(&input, &weight, 1, 1, 1, 5, 3, 1, 0, conv1d::F32);

    assert_eq!(result.len(), 3);
    assert_eq!(result, vec![-2.0, -2.0, -2.0]);
}

#[test]
fn test_conv1d_multi_channel_f32() {
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];

    let result: Vec<f32> = run_conv1d(&input, &weight, 1, 2, 1, 4, 2, 1, 0, conv1d::F32);

    assert_eq!(result.len(), 3);
    assert_eq!(result, vec![7.0, 9.0, 11.0]);
}

#[test]
fn test_conv2d_simple_f32() {
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let weight: Vec<f32> = vec![0.25, 0.25, 0.25, 0.25];

    let result: Vec<f32> = run_conv2d(&input, &weight, 1, 1, 1, 3, 3, 2, 2, 1, 1, 0, 0, conv2d::F32);

    assert_eq!(result.len(), 4);
    assert_eq!(result, vec![3.0, 4.0, 6.0, 7.0]);
}

#[test]
fn test_conv2d_stride_f32() {
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];

    let result: Vec<f32> = run_conv2d(&input, &weight, 1, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0, conv2d::F32);

    assert_eq!(result.len(), 4);
    assert_eq!(result, vec![7.0, 11.0, 23.0, 27.0]);
}

#[test]
fn test_conv3d_simple_f32() {
    // 2x2x2 cube
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // 2x2x2 kernel averaging all elements
    let weight: Vec<f32> = vec![0.125; 8];

    let result: Vec<f32> = run_conv3d(
        &input,
        &weight,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        0,
        0,
        0,
        conv3d::F32,
    );

    assert_eq!(result.len(), 1);
    // Average of 1..8 = 36/8 = 4.5
    assert_eq!(result, vec![4.5]);
}

#[test]
fn test_conv3d_stride_f32() {
    // 4x4x4 cube (64 elements)
    let input: Vec<f32> = (1..=64).map(|x| x as f32).collect();
    // 2x2x2 kernel (identity-like)
    let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let result: Vec<f32> = run_conv3d(
        &input,
        &weight,
        1,
        1,
        1,
        4,
        4,
        4,
        2,
        2,
        2,
        2,
        2,
        2,
        0,
        0,
        0,
        conv3d::F32,
    );

    // Output shape: ((4-2)/2+1, (4-2)/2+1, (4-2)/2+1) = (2, 2, 2) = 8 elements
    assert_eq!(result.len(), 8);
    // With stride 2, picking first element from each 2x2x2 window
    assert_eq!(result, vec![1.0, 3.0, 9.0, 11.0, 33.0, 35.0, 41.0, 43.0]);
}

#[test]
fn test_conv3d_padding_f32() {
    // 2x2x2 cube
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // 3x3x3 kernel (all 1s)
    let weight: Vec<f32> = vec![1.0; 27];

    let result: Vec<f32> = run_conv3d(
        &input,
        &weight,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
        1,
        1,
        1,
        1,
        1,
        1,
        conv3d::F32,
    );

    // Output shape with padding=1: ((2+2*1-3)/1+1)^3 = 2^3 = 8
    assert_eq!(result.len(), 8);
}

#[test]
fn test_conv3d_multi_channel_f32() {
    // 2 input channels, 2x2x2 each
    let input: Vec<f32> = vec![
        // Channel 0
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // Channel 1
        8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    ];
    // 2 output channels, each has 2 input channels * 2x2x2 kernel = 16 weights
    let weight: Vec<f32> = vec![
        // Out channel 0, in channel 0
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, // Out channel 0, in channel 1
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, // Out channel 1, in channel 0
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Out channel 1, in channel 1
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let result: Vec<f32> = run_conv3d(
        &input,
        &weight,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        0,
        0,
        0,
        conv3d::F32,
    );

    // Output: 2 channels, 1x1x1 each = 2 elements
    assert_eq!(result.len(), 2);
    // Channel 0: average of all inputs = (36 + 36)/2 = 36
    // Channel 1: first + last = 1 + 1 = 2
    assert_eq!(result, vec![36.0, 2.0]);
}

#[test]
fn test_conv3d_batch_f32() {
    // Batch of 2, 1x1x1 input each
    let input: Vec<f32> = vec![2.0, 3.0];
    // 1x1x1 kernel
    let weight: Vec<f32> = vec![4.0];

    let result: Vec<f32> = run_conv3d(
        &input,
        &weight,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        conv3d::F32,
    );

    assert_eq!(result.len(), 2);
    assert_eq!(result, vec![8.0, 12.0]);
}

#[test]
fn test_conv_transpose1d_simple_f32() {
    // Input: [1, 2]
    // Weight: [1, 2]
    // stride=2, padding=0, output_padding=0
    let input: Vec<f32> = vec![1.0, 2.0];
    let weight: Vec<f32> = vec![1.0, 2.0];

    let result: Vec<f32> = run_conv_transpose1d(&input, &weight, 1, 1, 1, 2, 2, 2, 0, 0, conv_transpose1d::F32);

    // Output length: (2-1)*2 - 2*0 + 2 + 0 = 4
    assert_eq!(result.len(), 4);
}

#[test]
fn test_conv_transpose2d_simple_f32() {
    // 2x2 input
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    // 2x2 kernel
    let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];

    let result: Vec<f32> = run_conv_transpose2d(
        &input,
        &weight,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        0,
        0,
        0,
        0,
        conv_transpose2d::F32,
    );

    // Output: (2-1)*2 - 0 + 2 + 0 = 4x4
    assert_eq!(result.len(), 16);
}

#[test]
fn test_conv_transpose3d_simple_f32() {
    // 1x1x1 input
    let input: Vec<f32> = vec![2.0];
    // 2x2x2 kernel (all 1s)
    let weight: Vec<f32> = vec![1.0; 8];

    let result: Vec<f32> = run_conv_transpose3d(
        &input,
        &weight,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        conv_transpose3d::F32,
    );

    // Output: (1-1)*1 + 2 = 2x2x2
    assert_eq!(result.len(), 8);
    // All elements should be 2.0 (input value * 1.0 weight)
    assert_eq!(result, vec![2.0; 8]);
}

#[test]
fn test_conv_transpose3d_stride_f32() {
    // 2x2x2 input
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // 2x2x2 kernel (identity-like)
    let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let result: Vec<f32> = run_conv_transpose3d(
        &input,
        &weight,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        conv_transpose3d::F32,
    );

    // Output: (2-1)*2 + 2 = 4x4x4
    assert_eq!(result.len(), 64);
}

#[test]
fn test_conv_transpose3d_padding_f32() {
    // 2x2x2 input
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // 3x3x3 kernel (center element is 1)
    let mut weight = vec![0.0; 27];
    weight[13] = 1.0; // Center of 3x3x3

    let result: Vec<f32> = run_conv_transpose3d(
        &input,
        &weight,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        conv_transpose3d::F32,
    );

    // Output: (2-1)*1 - 2*1 + 3 = 2x2x2
    assert_eq!(result.len(), 8);
}

#[test]
fn test_conv_transpose3d_output_padding_f32() {
    // 1x1x1 input
    let input: Vec<f32> = vec![3.0];
    // 2x2x2 kernel (all 1s)
    let weight: Vec<f32> = vec![1.0; 8];

    let result: Vec<f32> = run_conv_transpose3d(
        &input,
        &weight,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        0,
        0,
        0,
        1,
        1,
        1,
        conv_transpose3d::F32,
    );

    // Output: (1-1)*2 + 2 + 1 = 3x3x3
    assert_eq!(result.len(), 27);
}

#[test]
fn test_conv_transpose3d_multi_channel_f32() {
    // 2 input channels, 1x1x1 each
    let input: Vec<f32> = vec![2.0, 3.0];
    // Weight layout: (in_channels, out_channels, kernel_depth, kernel_height, kernel_width)
    // in_channels=2, out_channels=2, kernel=2x2x2
    let weight: Vec<f32> = vec![
        // In channel 0, out channel 0
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // In channel 0, out channel 1
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, // In channel 1, out channel 0
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // In channel 1, out channel 1
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ];

    let result: Vec<f32> = run_conv_transpose3d(
        &input,
        &weight,
        1,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        conv_transpose3d::F32,
    );

    // Output: 2 channels, 2x2x2 each = 16 elements
    assert_eq!(result.len(), 16);
    // Out channel 0: in_ch0(2.0)*1.0 + in_ch1(3.0)*1.0 = 5.0
    // Out channel 1: in_ch0(2.0)*0.5 + in_ch1(3.0)*0.5 = 2.5
    assert_eq!(result[..8], vec![5.0; 8]);
    assert_eq!(result[8..], vec![2.5; 8]);
}

#[test]
fn test_conv_transpose3d_batch_f32() {
    // Batch of 2, 1x1x1 input each
    let input: Vec<f32> = vec![1.0, 4.0];
    // 2x2x2 kernel (all 2s)
    let weight: Vec<f32> = vec![2.0; 8];

    let result: Vec<f32> = run_conv_transpose3d(
        &input,
        &weight,
        2,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        conv_transpose3d::F32,
    );

    // Output: batch=2, 2x2x2 each = 16 elements
    assert_eq!(result.len(), 16);
    // Batch 0: all 2.0 (1*2), Batch 1: all 8.0 (4*2)
    assert_eq!(result[..8], vec![2.0; 8]);
    assert_eq!(result[8..], vec![8.0; 8]);
}

#[test]
fn test_conv1d_grad_weight_f32() {
    // Input: [1, 2, 3]
    let input: Vec<f32> = vec![1.0, 2.0, 3.0];
    // Grad output: [1, 1]
    let grad_output: Vec<f32> = vec![1.0, 1.0];

    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input_buffer = new_buffer(&device, &input);
    let grad_output_buffer = new_buffer(&device, &grad_output);

    // Weight gradient shape: [out_channels=1, in_channels=1, kernel_size=2]
    let grad_weight_size = 1 * 1 * 2;
    let grad_weight = new_buffer_zeroed::<f32>(&device, grad_weight_size);

    // Generic metadata layout for conv1d_grad_weight (input_ndim=3, spatial_dims=1):
    // [num_els, input_ndim, spatial_dims,
    //  input_shape(3), grad_output_shape(3), weight_shape(3),
    //  input_strides(3), grad_output_strides(3),
    //  input_offset, grad_output_offset,
    //  stride, padding, dilation]
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_width = 3;
    let out_width = 2;
    let kernel_width = 2;
    let metadata = vec![
        grad_weight_size, // num_els
        3,                // input_ndim
        1,                // spatial_dims
        // input_shape: [batch, in_channels, in_width]
        batch,
        in_channels,
        in_width,
        // grad_output_shape: [batch, out_channels, out_width]
        batch,
        out_channels,
        out_width,
        // weight_shape: [out_channels, in_channels, kernel_width]
        out_channels,
        in_channels,
        kernel_width,
        // input_strides: [in_channels * in_width, in_width, 1]
        in_channels * in_width,
        in_width,
        1,
        // grad_output_strides: [out_channels * out_width, out_width, 1]
        out_channels * out_width,
        out_width,
        1,
        // input_offset, grad_output_offset
        0,
        0,
        // stride, padding, dilation
        1,
        0,
        1,
    ];

    call_ops_conv_grad_weight(
        conv1d_grad_weight::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&grad_output_buffer),
        &grad_weight,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result: Vec<f32> = read_to_vec(&grad_weight, grad_weight_size);
    assert_eq!(result.len(), 2);
    // Grad weight should be computed from input and grad_output
    // Position 0: 1*1 + 2*1 = 3
    // Position 1: 2*1 + 3*1 = 5
    assert_eq!(result, vec![3.0, 5.0]);
}

#[test]
fn test_conv2d_grad_weight_f32() {
    // 2x2 input
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    // 1x1 grad output (from 2x2 kernel, stride 1)
    let grad_output: Vec<f32> = vec![1.0];

    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input_buffer = new_buffer(&device, &input);
    let grad_output_buffer = new_buffer(&device, &grad_output);

    // Weight gradient shape: [1, 1, 2, 2]
    let grad_weight_size = 4;
    let grad_weight = new_buffer_zeroed::<f32>(&device, grad_weight_size);

    // Generic metadata layout for conv2d_grad_weight (input_ndim=4, spatial_dims=2):
    // [num_els, input_ndim, spatial_dims,
    //  input_shape(4), grad_output_shape(4), weight_shape(4),
    //  input_strides(4), grad_output_strides(4),
    //  input_offset, grad_output_offset,
    //  stride(2), padding(2), dilation(2)]
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_height = 2;
    let in_width = 2;
    let out_height = 1;
    let out_width = 1;
    let kernel_height = 2;
    let kernel_width = 2;
    let metadata = vec![
        grad_weight_size, // num_els
        4,                // input_ndim
        2,                // spatial_dims
        // input_shape: [batch, in_channels, in_height, in_width]
        batch,
        in_channels,
        in_height,
        in_width,
        // grad_output_shape: [batch, out_channels, out_height, out_width]
        batch,
        out_channels,
        out_height,
        out_width,
        // weight_shape: [out_channels, in_channels, kernel_height, kernel_width]
        out_channels,
        in_channels,
        kernel_height,
        kernel_width,
        // input_strides: [C*H*W, H*W, W, 1]
        in_channels * in_height * in_width,
        in_height * in_width,
        in_width,
        1,
        // grad_output_strides: [C*H*W, H*W, W, 1]
        out_channels * out_height * out_width,
        out_height * out_width,
        out_width,
        1,
        // input_offset, grad_output_offset
        0,
        0,
        // stride_h, stride_w
        1,
        1,
        // padding_h, padding_w
        0,
        0,
        // dilation_h, dilation_w
        1,
        1,
    ];

    call_ops_conv_grad_weight(
        conv2d_grad_weight::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&grad_output_buffer),
        &grad_weight,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result: Vec<f32> = read_to_vec(&grad_weight, grad_weight_size);
    assert_eq!(result.len(), 4);
    // Grad weight should be the input itself since grad_output is [1.0]
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_conv3d_grad_weight_f32() {
    // 2x2x2 input
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // 1x1x1 grad output (from 2x2x2 kernel, stride 1)
    let grad_output: Vec<f32> = vec![1.0];

    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input_buffer = new_buffer(&device, &input);
    let grad_output_buffer = new_buffer(&device, &grad_output);

    // Weight gradient shape: [1, 1, 2, 2, 2]
    let grad_weight_size = 8;
    let grad_weight = new_buffer_zeroed::<f32>(&device, grad_weight_size);

    // Generic metadata layout for conv3d_grad_weight (input_ndim=5, spatial_dims=3):
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_depth = 2;
    let in_height = 2;
    let in_width = 2;
    let out_depth = 1;
    let out_height = 1;
    let out_width = 1;
    let kernel_depth = 2;
    let kernel_height = 2;
    let kernel_width = 2;
    let metadata = vec![
        grad_weight_size, // num_els
        5,                // input_ndim
        3,                // spatial_dims
        // input_shape: [batch, in_channels, in_depth, in_height, in_width]
        batch,
        in_channels,
        in_depth,
        in_height,
        in_width,
        // grad_output_shape: [batch, out_channels, out_depth, out_height, out_width]
        batch,
        out_channels,
        out_depth,
        out_height,
        out_width,
        // weight_shape: [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]
        out_channels,
        in_channels,
        kernel_depth,
        kernel_height,
        kernel_width,
        // input_strides: [C*D*H*W, D*H*W, H*W, W, 1]
        in_channels * in_depth * in_height * in_width,
        in_depth * in_height * in_width,
        in_height * in_width,
        in_width,
        1,
        // grad_output_strides: [C*D*H*W, D*H*W, H*W, W, 1]
        out_channels * out_depth * out_height * out_width,
        out_depth * out_height * out_width,
        out_height * out_width,
        out_width,
        1,
        // input_offset, grad_output_offset
        0,
        0,
        // stride_d, stride_h, stride_w
        1,
        1,
        1,
        // padding_d, padding_h, padding_w
        0,
        0,
        0,
        // dilation_d, dilation_h, dilation_w
        1,
        1,
        1,
    ];

    call_ops_conv_grad_weight(
        conv3d_grad_weight::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&grad_output_buffer),
        &grad_weight,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result: Vec<f32> = read_to_vec(&grad_weight, grad_weight_size);
    assert_eq!(result.len(), 8);
    // Grad weight should be the input itself since grad_output is [1.0]
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_conv3d_grad_weight_multi_batch_f32() {
    // Batch=2, 2x2x2 input each
    let input: Vec<f32> = vec![
        // Batch 0
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // Batch 1
        8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    ];
    // 1x1x1 grad output per batch
    let grad_output: Vec<f32> = vec![1.0, 2.0];

    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input_buffer = new_buffer(&device, &input);
    let grad_output_buffer = new_buffer(&device, &grad_output);

    // Weight gradient shape: [1, 1, 2, 2, 2]
    let grad_weight_size = 8;
    let grad_weight = new_buffer_zeroed::<f32>(&device, grad_weight_size);

    // Generic metadata layout for conv3d_grad_weight (input_ndim=5, spatial_dims=3) with batch=2
    let batch = 2;
    let in_channels = 1;
    let out_channels = 1;
    let in_depth = 2;
    let in_height = 2;
    let in_width = 2;
    let out_depth = 1;
    let out_height = 1;
    let out_width = 1;
    let kernel_depth = 2;
    let kernel_height = 2;
    let kernel_width = 2;
    let metadata = vec![
        grad_weight_size, // num_els
        5,                // input_ndim
        3,                // spatial_dims
        // input_shape
        batch,
        in_channels,
        in_depth,
        in_height,
        in_width,
        // grad_output_shape
        batch,
        out_channels,
        out_depth,
        out_height,
        out_width,
        // weight_shape
        out_channels,
        in_channels,
        kernel_depth,
        kernel_height,
        kernel_width,
        // input_strides
        in_channels * in_depth * in_height * in_width,
        in_depth * in_height * in_width,
        in_height * in_width,
        in_width,
        1,
        // grad_output_strides
        out_channels * out_depth * out_height * out_width,
        out_depth * out_height * out_width,
        out_height * out_width,
        out_width,
        1,
        // input_offset, grad_output_offset
        0,
        0,
        // stride, padding, dilation (3 each)
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
    ];

    call_ops_conv_grad_weight(
        conv3d_grad_weight::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&grad_output_buffer),
        &grad_weight,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result: Vec<f32> = read_to_vec(&grad_weight, grad_weight_size);
    assert_eq!(result.len(), 8);
    // Grad weight accumulates: batch0 * 1.0 + batch1 * 2.0
    // [1*1 + 8*2, 2*1 + 7*2, 3*1 + 6*2, 4*1 + 5*2, 5*1 + 4*2, 6*1 + 3*2, 7*1 + 2*2, 8*1 + 1*2]
    assert_eq!(result, vec![17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0]);
}

#[test]
fn test_conv_transpose1d_grad_weight_f32() {
    // Input: [1, 2, 3]
    let input: Vec<f32> = vec![1.0, 2.0, 3.0];
    // Grad output: [1, 1, 1, 1] (4 elements from transpose conv)
    let grad_output: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input_buffer = new_buffer(&device, &input);
    let grad_output_buffer = new_buffer(&device, &grad_output);

    // Weight gradient shape: [out_channels=1, in_channels=1, kernel_size=2]
    let grad_weight_size = 2;
    let grad_weight = new_buffer_zeroed::<f32>(&device, grad_weight_size);

    // Generic metadata layout for conv_transpose1d_grad_weight (input_ndim=3, spatial_dims=1):
    let batch = 1;
    let in_channels = 1; // Note: for transpose conv, this is the input to transpose conv (which is out_channels of forward)
    let out_channels = 1;
    let in_width = 3;
    let out_width = 4;
    let kernel_width = 2;
    let metadata = vec![
        grad_weight_size, // num_els
        3,                // input_ndim
        1,                // spatial_dims
        // input_shape: [batch, in_channels, in_width]
        batch,
        in_channels,
        in_width,
        // grad_output_shape: [batch, out_channels, out_width]
        batch,
        out_channels,
        out_width,
        // weight_shape: [in_channels, out_channels, kernel_width] (transposed)
        in_channels,
        out_channels,
        kernel_width,
        // input_strides
        in_channels * in_width,
        in_width,
        1,
        // grad_output_strides
        out_channels * out_width,
        out_width,
        1,
        // input_offset, grad_output_offset
        0,
        0,
        // stride, padding, dilation
        1,
        0,
        1,
    ];

    call_ops_conv_grad_weight(
        conv_transpose1d_grad_weight::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&grad_output_buffer),
        &grad_weight,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result: Vec<f32> = read_to_vec(&grad_weight, grad_weight_size);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_conv_transpose2d_grad_weight_f32() {
    // 2x2 input
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    // 4x4 grad output (from transpose conv with stride 2)
    let grad_output: Vec<f32> = vec![1.0; 16];

    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input_buffer = new_buffer(&device, &input);
    let grad_output_buffer = new_buffer(&device, &grad_output);

    // Weight gradient shape: [1, 1, 2, 2]
    let grad_weight_size = 4;
    let grad_weight = new_buffer_zeroed::<f32>(&device, grad_weight_size);

    // Generic metadata layout for conv_transpose2d_grad_weight (input_ndim=4, spatial_dims=2):
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_height = 2;
    let in_width = 2;
    let out_height = 4;
    let out_width = 4;
    let kernel_height = 2;
    let kernel_width = 2;
    let metadata = vec![
        grad_weight_size, // num_els
        4,                // input_ndim
        2,                // spatial_dims
        // input_shape
        batch,
        in_channels,
        in_height,
        in_width,
        // grad_output_shape
        batch,
        out_channels,
        out_height,
        out_width,
        // weight_shape (transposed: [in_channels, out_channels, kH, kW])
        in_channels,
        out_channels,
        kernel_height,
        kernel_width,
        // input_strides
        in_channels * in_height * in_width,
        in_height * in_width,
        in_width,
        1,
        // grad_output_strides
        out_channels * out_height * out_width,
        out_height * out_width,
        out_width,
        1,
        // input_offset, grad_output_offset
        0,
        0,
        // stride_h, stride_w
        2,
        2,
        // padding_h, padding_w
        0,
        0,
        // dilation_h, dilation_w
        1,
        1,
    ];

    call_ops_conv_grad_weight(
        conv_transpose2d_grad_weight::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&grad_output_buffer),
        &grad_weight,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result: Vec<f32> = read_to_vec(&grad_weight, grad_weight_size);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_conv_transpose2d_grad_weight_multi_batch_f32() {
    // Batch=2, 2x2 input each
    let input: Vec<f32> = vec![
        // Batch 0
        1.0, 2.0, 3.0, 4.0, // Batch 1
        4.0, 3.0, 2.0, 1.0,
    ];
    // 4x4 grad output per batch
    let grad_output: Vec<f32> = vec![
        // Batch 0: all 1.0
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // Batch 1: all 2.0
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];

    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input_buffer = new_buffer(&device, &input);
    let grad_output_buffer = new_buffer(&device, &grad_output);

    // Weight gradient shape: [1, 1, 2, 2]
    let grad_weight_size = 4;
    let grad_weight = new_buffer_zeroed::<f32>(&device, grad_weight_size);

    // Generic metadata layout with batch=2
    let batch = 2;
    let in_channels = 1;
    let out_channels = 1;
    let in_height = 2;
    let in_width = 2;
    let out_height = 4;
    let out_width = 4;
    let kernel_height = 2;
    let kernel_width = 2;
    let metadata = vec![
        grad_weight_size, // num_els
        4,                // input_ndim
        2,                // spatial_dims
        // input_shape
        batch,
        in_channels,
        in_height,
        in_width,
        // grad_output_shape
        batch,
        out_channels,
        out_height,
        out_width,
        // weight_shape
        in_channels,
        out_channels,
        kernel_height,
        kernel_width,
        // input_strides
        in_channels * in_height * in_width,
        in_height * in_width,
        in_width,
        1,
        // grad_output_strides
        out_channels * out_height * out_width,
        out_height * out_width,
        out_width,
        1,
        // input_offset, grad_output_offset
        0,
        0,
        // stride, padding, dilation
        2,
        2,
        0,
        0,
        1,
        1,
    ];

    call_ops_conv_grad_weight(
        conv_transpose2d_grad_weight::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&grad_output_buffer),
        &grad_weight,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result: Vec<f32> = read_to_vec(&grad_weight, grad_weight_size);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_conv_transpose3d_grad_weight_f32() {
    // 2x2x2 input
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // 2x2x2 grad output (from transpose conv with stride 1)
    let grad_output: Vec<f32> = vec![1.0; 8];

    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input_buffer = new_buffer(&device, &input);
    let grad_output_buffer = new_buffer(&device, &grad_output);

    // Weight gradient shape: [1, 1, 2, 2, 2]
    let grad_weight_size = 8;
    let grad_weight = new_buffer_zeroed::<f32>(&device, grad_weight_size);

    // Generic metadata layout for conv_transpose3d_grad_weight (input_ndim=5, spatial_dims=3):
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_depth = 2;
    let in_height = 2;
    let in_width = 2;
    let out_depth = 2;
    let out_height = 2;
    let out_width = 2;
    let kernel_depth = 2;
    let kernel_height = 2;
    let kernel_width = 2;
    let metadata = vec![
        grad_weight_size, // num_els
        5,                // input_ndim
        3,                // spatial_dims
        // input_shape
        batch,
        in_channels,
        in_depth,
        in_height,
        in_width,
        // grad_output_shape
        batch,
        out_channels,
        out_depth,
        out_height,
        out_width,
        // weight_shape
        in_channels,
        out_channels,
        kernel_depth,
        kernel_height,
        kernel_width,
        // input_strides
        in_channels * in_depth * in_height * in_width,
        in_depth * in_height * in_width,
        in_height * in_width,
        in_width,
        1,
        // grad_output_strides
        out_channels * out_depth * out_height * out_width,
        out_depth * out_height * out_width,
        out_height * out_width,
        out_width,
        1,
        // input_offset, grad_output_offset
        0,
        0,
        // stride, padding, dilation (3 each)
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
    ];

    call_ops_conv_grad_weight(
        conv_transpose3d_grad_weight::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&grad_output_buffer),
        &grad_weight,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result: Vec<f32> = read_to_vec(&grad_weight, grad_weight_size);
    assert_eq!(result.len(), 8);
}

#[test]
fn test_conv_transpose3d_grad_weight_multi_batch_f32() {
    // Batch=2, 1x1x1 input each
    let input: Vec<f32> = vec![2.0, 3.0];
    // 2x2x2 grad output per batch
    let grad_output: Vec<f32> = vec![
        // Batch 0: all 1.0
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // Batch 1: all 2.0
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];

    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input_buffer = new_buffer(&device, &input);
    let grad_output_buffer = new_buffer(&device, &grad_output);

    // Weight gradient shape: [1, 1, 2, 2, 2]
    let grad_weight_size = 8;
    let grad_weight = new_buffer_zeroed::<f32>(&device, grad_weight_size);

    // Generic metadata layout with batch=2
    let batch = 2;
    let in_channels = 1;
    let out_channels = 1;
    let in_depth = 1;
    let in_height = 1;
    let in_width = 1;
    let out_depth = 2;
    let out_height = 2;
    let out_width = 2;
    let kernel_depth = 2;
    let kernel_height = 2;
    let kernel_width = 2;
    let metadata = vec![
        grad_weight_size, // num_els
        5,                // input_ndim
        3,                // spatial_dims
        // input_shape
        batch,
        in_channels,
        in_depth,
        in_height,
        in_width,
        // grad_output_shape
        batch,
        out_channels,
        out_depth,
        out_height,
        out_width,
        // weight_shape
        in_channels,
        out_channels,
        kernel_depth,
        kernel_height,
        kernel_width,
        // input_strides
        in_channels * in_depth * in_height * in_width,
        in_depth * in_height * in_width,
        in_height * in_width,
        in_width,
        1,
        // grad_output_strides
        out_channels * out_depth * out_height * out_width,
        out_depth * out_height * out_width,
        out_height * out_width,
        out_width,
        1,
        // input_offset, grad_output_offset
        0,
        0,
        // stride, padding, dilation (3 each)
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
    ];

    call_ops_conv_grad_weight(
        conv_transpose3d_grad_weight::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&grad_output_buffer),
        &grad_weight,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result: Vec<f32> = read_to_vec(&grad_weight, grad_weight_size);
    assert_eq!(result.len(), 8);
    // Grad weight accumulates: batch0*1.0 + batch1*2.0
    // All elements: 2*1 + 3*2 = 8
    assert_eq!(result, vec![8.0; 8]);
}
