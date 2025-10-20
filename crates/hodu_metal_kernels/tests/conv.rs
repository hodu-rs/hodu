use half::{bf16, f16};
use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_conv, conv1d, conv2d, Kernel},
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
    name: Kernel,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;

    let input_buffer = new_buffer(&device, input);
    let weight_buffer = new_buffer(&device, weight);

    // Calculate output length
    let output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    let output_size = batch * out_channels * output_length;

    let output = device
        .new_buffer(output_size * std::mem::size_of::<T>(), options)
        .unwrap();

    // Metadata for conv1d: [num_els, batch, in_channels, out_channels, in_width, kernel_width, out_width, stride, padding, dilation, input_offset, weight_offset]
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
        1, // dilation
        0, // input_offset
        0, // weight_offset
    ];

    call_conv(
        &device,
        &command_buffer,
        &kernels,
        name,
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
    name: Kernel,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;

    let input_buffer = new_buffer(&device, input);
    let weight_buffer = new_buffer(&device, weight);

    // Calculate output dimensions
    let output_height = (input_height + 2 * padding_h - kernel_h) / stride_h + 1;
    let output_width = (input_width + 2 * padding_w - kernel_w) / stride_w + 1;
    let output_size = batch * out_channels * output_height * output_width;

    let output = device
        .new_buffer(output_size * std::mem::size_of::<T>(), options)
        .unwrap();

    // Metadata for conv2d: [num_els, batch, in_channels, out_channels,
    //                       in_height, in_width, kernel_height, kernel_width,
    //                       out_height, out_width,
    //                       stride_h, stride_w, padding_h, padding_w,
    //                       dilation_h, dilation_w, input_offset, weight_offset]
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
        1, // dilation_h
        1, // dilation_w
        0, // input_offset
        0, // weight_offset
    ];

    call_conv(
        &device,
        &command_buffer,
        &kernels,
        name,
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

#[test]
fn test_conv1d_simple_f32() {
    // Simple 1D convolution test
    // Input: [1, 2, 3, 4, 5] (batch=1, in_channels=1, length=5)
    // Weight: [1, 0, -1] (out_channels=1, in_channels=1, kernel_size=3)
    // Expected output: approximate edge detection
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight: Vec<f32> = vec![1.0, 0.0, -1.0];

    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let input_length = 5;
    let kernel_size = 3;
    let stride = 1;
    let padding = 0;

    let result: Vec<f32> = run_conv1d(
        &input,
        &weight,
        batch,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        conv1d::F32,
    );

    // Output length: (5 + 0 - 3) / 1 + 1 = 3
    // [1,2,3] * [1,0,-1] = 1*1 + 2*0 + 3*(-1) = -2
    // [2,3,4] * [1,0,-1] = 2*1 + 3*0 + 4*(-1) = -2
    // [3,4,5] * [1,0,-1] = 3*1 + 4*0 + 5*(-1) = -2
    assert_eq!(result.len(), 3);
    assert_eq!(result, vec![-2.0, -2.0, -2.0]);
}

#[test]
fn test_conv1d_multi_channel_f32() {
    // Multi-channel 1D convolution
    // Input: batch=1, in_channels=2, length=4
    // Weight: out_channels=1, in_channels=2, kernel_size=2
    let input: Vec<f32> = vec![
        // Channel 0
        1.0, 2.0, 3.0, 4.0, // Channel 1
        5.0, 6.0, 7.0, 8.0,
    ];
    let weight: Vec<f32> = vec![
        // Output channel 0, Input channel 0
        0.5, 0.5, // Output channel 0, Input channel 1
        0.5, 0.5,
    ];

    let batch = 1;
    let in_channels = 2;
    let out_channels = 1;
    let input_length = 4;
    let kernel_size = 2;
    let stride = 1;
    let padding = 0;

    let result: Vec<f32> = run_conv1d(
        &input,
        &weight,
        batch,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        conv1d::F32,
    );

    // Output length: (4 + 0 - 2) / 1 + 1 = 3
    assert_eq!(result.len(), 3);

    // Expected results (approximate due to multi-channel summation):
    // Position 0: (1+2)*0.5 + (5+6)*0.5 = 1.5 + 5.5 = 7.0
    // Position 1: (2+3)*0.5 + (6+7)*0.5 = 2.5 + 6.5 = 9.0
    // Position 2: (3+4)*0.5 + (7+8)*0.5 = 3.5 + 7.5 = 11.0
    assert_eq!(result, vec![7.0, 9.0, 11.0]);
}

#[test]
fn test_conv2d_simple_f32() {
    // Simple 2D convolution test (identity kernel)
    // Input: 3x3 image, single channel
    // Weight: 2x2 kernel averaging
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let weight: Vec<f32> = vec![0.25, 0.25, 0.25, 0.25];

    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let input_height = 3;
    let input_width = 3;
    let kernel_h = 2;
    let kernel_w = 2;
    let stride_h = 1;
    let stride_w = 1;
    let padding_h = 0;
    let padding_w = 0;

    let result: Vec<f32> = run_conv2d(
        &input,
        &weight,
        batch,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        conv2d::F32,
    );

    // Output size: 2x2
    assert_eq!(result.len(), 4);

    // Expected (averaging 2x2 windows):
    // Top-left: (1+2+4+5)/4 = 3.0
    // Top-right: (2+3+5+6)/4 = 4.0
    // Bottom-left: (4+5+7+8)/4 = 6.0
    // Bottom-right: (5+6+8+9)/4 = 7.0
    assert_eq!(result, vec![3.0, 4.0, 6.0, 7.0]);
}

#[test]
fn test_conv1d_bf16() {
    let input: Vec<bf16> = vec![1.0, 2.0, 3.0, 4.0].into_iter().map(bf16::from_f32).collect();
    let weight: Vec<bf16> = vec![1.0, 1.0].into_iter().map(bf16::from_f32).collect();

    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let input_length = 4;
    let kernel_size = 2;
    let stride = 1;
    let padding = 0;

    let result: Vec<bf16> = run_conv1d(
        &input,
        &weight,
        batch,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        conv1d::BF16,
    );

    let expected: Vec<bf16> = vec![3.0, 5.0, 7.0].into_iter().map(bf16::from_f32).collect();

    assert_eq!(result.len(), 3);
    assert_eq!(result, expected);
}

#[test]
fn test_conv1d_f16() {
    let input: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0].into_iter().map(f16::from_f32).collect();
    let weight: Vec<f16> = vec![1.0, 1.0].into_iter().map(f16::from_f32).collect();

    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let input_length = 4;
    let kernel_size = 2;
    let stride = 1;
    let padding = 0;

    let result: Vec<f16> = run_conv1d(
        &input,
        &weight,
        batch,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        conv1d::F16,
    );

    let expected: Vec<f16> = vec![3.0, 5.0, 7.0].into_iter().map(f16::from_f32).collect();

    assert_eq!(result.len(), 3);
    assert_eq!(result, expected);
}

#[test]
fn test_conv2d_stride_f32() {
    // Test 2D convolution with stride > 1
    // Input: 4x4 image
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 kernel summing corners

    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let input_height = 4;
    let input_width = 4;
    let kernel_h = 2;
    let kernel_w = 2;
    let stride_h = 2;
    let stride_w = 2;
    let padding_h = 0;
    let padding_w = 0;

    let result: Vec<f32> = run_conv2d(
        &input,
        &weight,
        batch,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        conv2d::F32,
    );

    // Output size: 2x2 (with stride 2)
    assert_eq!(result.len(), 4);

    // Expected (summing top-left and bottom-right of 2x2 windows):
    // Top-left window: 1 + 6 = 7
    // Top-right window: 3 + 8 = 11
    // Bottom-left window: 9 + 14 = 23
    // Bottom-right window: 11 + 16 = 27
    assert_eq!(result, vec![7.0, 11.0, 23.0, 27.0]);
}
