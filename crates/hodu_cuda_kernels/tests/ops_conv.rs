use hodu_cuda_kernels::{kernel::Kernels, kernels::*};

fn device() -> Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

fn kernels() -> Kernels {
    Kernels::new()
}

#[allow(clippy::too_many_arguments)]
fn run_conv1d<T: cudarc::driver::DeviceRepr + Clone>(
    input: &[T],
    weight: &[T],
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    input_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    kernel: hodu_cuda_kernels::kernels::Kernel,
) -> Vec<T> {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let weight_dev = stream.memcpy_stod(weight).unwrap();

    let output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    let output_size = batch * out_channels * output_length;

    let mut output: cudarc::driver::CudaSlice<T> = unsafe { stream.alloc(output_size).unwrap() };

    // Metadata: [output_size, batch, in_channels, out_channels, input_length, kernel_size,
    //            output_length, stride, padding, dilation, input_offset, weight_offset]
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

    call_ops_conv(
        kernel,
        &kernels,
        &device,
        &input_dev,
        &weight_dev,
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![unsafe { core::mem::zeroed() }; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    results
}

#[allow(clippy::too_many_arguments)]
fn run_conv2d<T: cudarc::driver::DeviceRepr + Clone>(
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
    kernel: hodu_cuda_kernels::kernels::Kernel,
) -> Vec<T> {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let weight_dev = stream.memcpy_stod(weight).unwrap();

    let output_height = (input_height + 2 * padding_h - kernel_h) / stride_h + 1;
    let output_width = (input_width + 2 * padding_w - kernel_w) / stride_w + 1;
    let output_size = batch * out_channels * output_height * output_width;

    let mut output: cudarc::driver::CudaSlice<T> = unsafe { stream.alloc(output_size).unwrap() };

    // Metadata: [output_size, batch, in_channels, out_channels, input_height, input_width,
    //            kernel_h, kernel_w, output_height, output_width, stride_h, stride_w,
    //            padding_h, padding_w, dilation_h, dilation_w, input_offset, weight_offset]
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

    call_ops_conv(
        kernel,
        &kernels,
        &device,
        &input_dev,
        &weight_dev,
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![unsafe { core::mem::zeroed() }; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    results
}

#[allow(clippy::too_many_arguments)]
fn run_conv3d<T: cudarc::driver::DeviceRepr + Clone>(
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
    kernel: hodu_cuda_kernels::kernels::Kernel,
) -> Vec<T> {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let weight_dev = stream.memcpy_stod(weight).unwrap();

    let output_depth = (input_depth + 2 * padding_d - kernel_d) / stride_d + 1;
    let output_height = (input_height + 2 * padding_h - kernel_h) / stride_h + 1;
    let output_width = (input_width + 2 * padding_w - kernel_w) / stride_w + 1;
    let output_size = batch * out_channels * output_depth * output_height * output_width;

    let mut output: cudarc::driver::CudaSlice<T> = unsafe { stream.alloc(output_size).unwrap() };

    // Metadata: [output_size, batch, in_channels, out_channels, input_depth, input_height, input_width,
    //            kernel_d, kernel_h, kernel_w, output_depth, output_height, output_width,
    //            stride_d, stride_h, stride_w, padding_d, padding_h, padding_w,
    //            dilation_d, dilation_h, dilation_w, input_offset, weight_offset]
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
        1, // dilation_d
        1, // dilation_h
        1, // dilation_w
        0, // input_offset
        0, // weight_offset
    ];

    call_ops_conv(
        kernel,
        &kernels,
        &device,
        &input_dev,
        &weight_dev,
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![unsafe { core::mem::zeroed() }; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    results
}

#[allow(clippy::too_many_arguments)]
fn run_conv_transpose1d<T: cudarc::driver::DeviceRepr + Clone>(
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
    kernel: hodu_cuda_kernels::kernels::Kernel,
) -> Vec<T> {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let weight_dev = stream.memcpy_stod(weight).unwrap();

    let output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    let output_size = batch * out_channels * output_length;

    let mut output: cudarc::driver::CudaSlice<T> = unsafe { stream.alloc(output_size).unwrap() };

    // Metadata: [output_size, batch, in_channels, out_channels, input_length, kernel_size,
    //            output_length, stride, padding, dilation, output_padding, input_offset, weight_offset]
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
        output_padding,
        0, // input_offset
        0, // weight_offset
    ];

    call_ops_conv(
        kernel,
        &kernels,
        &device,
        &input_dev,
        &weight_dev,
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![unsafe { core::mem::zeroed() }; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    results
}

#[allow(clippy::too_many_arguments)]
fn run_conv_transpose2d<T: cudarc::driver::DeviceRepr + Clone>(
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
    kernel: hodu_cuda_kernels::kernels::Kernel,
) -> Vec<T> {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let weight_dev = stream.memcpy_stod(weight).unwrap();

    let output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    let output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;
    let output_size = batch * out_channels * output_height * output_width;

    let mut output: cudarc::driver::CudaSlice<T> = unsafe { stream.alloc(output_size).unwrap() };

    // Metadata: [output_size, batch, in_channels, out_channels, input_height, input_width,
    //            kernel_h, kernel_w, output_height, output_width, stride_h, stride_w,
    //            padding_h, padding_w, dilation_h, dilation_w, output_padding_h, output_padding_w,
    //            input_offset, weight_offset]
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
        output_padding_h,
        output_padding_w,
        0, // input_offset
        0, // weight_offset
    ];

    call_ops_conv(
        kernel,
        &kernels,
        &device,
        &input_dev,
        &weight_dev,
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![unsafe { core::mem::zeroed() }; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    results
}

#[allow(clippy::too_many_arguments)]
fn run_conv_transpose3d<T: cudarc::driver::DeviceRepr + Clone>(
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
    kernel: hodu_cuda_kernels::kernels::Kernel,
) -> Vec<T> {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let weight_dev = stream.memcpy_stod(weight).unwrap();

    let output_depth = (input_depth - 1) * stride_d + kernel_d - 2 * padding_d + output_padding_d;
    let output_height = (input_height - 1) * stride_h + kernel_h - 2 * padding_h + output_padding_h;
    let output_width = (input_width - 1) * stride_w + kernel_w - 2 * padding_w + output_padding_w;
    let output_size = batch * out_channels * output_depth * output_height * output_width;

    let mut output: cudarc::driver::CudaSlice<T> = unsafe { stream.alloc(output_size).unwrap() };

    // Metadata: [output_size, batch, in_channels, out_channels, input_depth, input_height, input_width,
    //            kernel_d, kernel_h, kernel_w, output_depth, output_height, output_width,
    //            stride_d, stride_h, stride_w, padding_d, padding_h, padding_w,
    //            dilation_d, dilation_h, dilation_w, output_padding_d, output_padding_h, output_padding_w,
    //            input_offset, weight_offset]
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
        1, // dilation_d
        1, // dilation_h
        1, // dilation_w
        output_padding_d,
        output_padding_h,
        output_padding_w,
        0, // input_offset
        0, // weight_offset
    ];

    call_ops_conv(
        kernel,
        &kernels,
        &device,
        &input_dev,
        &weight_dev,
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![unsafe { core::mem::zeroed() }; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    results
}

#[test]
fn test_conv1d_simple_f32() {
    // Input: [1, 2, 3, 4, 5]
    // Weight: [1, 0, -1] (edge detector)
    // stride=1, padding=0
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight: Vec<f32> = vec![1.0, 0.0, -1.0];

    let result: Vec<f32> = run_conv1d(&input, &weight, 1, 1, 1, 5, 3, 1, 0, conv1d::F32);

    assert_eq!(result.len(), 3);
    // Windows: [1,2,3] -> 1*1 + 2*0 + 3*(-1) = -2
    //          [2,3,4] -> 2*1 + 3*0 + 4*(-1) = -2
    //          [3,4,5] -> 3*1 + 4*0 + 5*(-1) = -2
    assert_eq!(result, vec![-2.0, -2.0, -2.0]);
}

#[test]
fn test_conv1d_multi_channel_f32() {
    // Input: 2 channels, 4 elements each
    // Input shape: [batch=1, in_channels=2, length=4]
    // Input layout: [ch0: 1,2,3,4, ch1: 5,6,7,8]
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // Weight: [out_channels=1, in_channels=2, kernel_size=2]
    // Weight layout: [ch0: 0.5,0.5, ch1: 0.5,0.5]
    let weight: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];

    let result: Vec<f32> = run_conv1d(&input, &weight, 1, 2, 1, 4, 2, 1, 0, conv1d::F32);

    assert_eq!(result.len(), 3);
    // Windows for channel 0: [1,2], [2,3], [3,4]
    // Windows for channel 1: [5,6], [6,7], [7,8]
    // Output[0] = (1+2)*0.5 + (5+6)*0.5 = 1.5 + 5.5 = 7.0
    // Output[1] = (2+3)*0.5 + (6+7)*0.5 = 2.5 + 6.5 = 9.0
    // Output[2] = (3+4)*0.5 + (7+8)*0.5 = 3.5 + 7.5 = 11.0
    assert_eq!(result, vec![7.0, 9.0, 11.0]);
}

#[test]
fn test_conv2d_simple_f32() {
    // Input: 3x3 matrix
    // [[1, 2, 3],
    //  [4, 5, 6],
    //  [7, 8, 9]]
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    // Weight: 2x2 averaging filter
    let weight: Vec<f32> = vec![0.25, 0.25, 0.25, 0.25];

    let result: Vec<f32> = run_conv2d(&input, &weight, 1, 1, 1, 3, 3, 2, 2, 1, 1, 0, 0, conv2d::F32);

    assert_eq!(result.len(), 4);
    // Windows: [[1,2,4,5], [2,3,5,6], [4,5,7,8], [5,6,8,9]]
    // Average of [1,2,4,5] = 3.0
    // Average of [2,3,5,6] = 4.0
    // Average of [4,5,7,8] = 6.0
    // Average of [5,6,8,9] = 7.0
    assert_eq!(result, vec![3.0, 4.0, 6.0, 7.0]);
}

#[test]
fn test_conv2d_stride_f32() {
    // Input: 4x4 matrix
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    // Weight: 2x2 picking first and last diagonal elements
    let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];

    let result: Vec<f32> = run_conv2d(&input, &weight, 1, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0, conv2d::F32);

    assert_eq!(result.len(), 4);
    // With stride 2: [[1,2,5,6], [3,4,7,8], [9,10,13,14], [11,12,15,16]]
    // [[1,2,5,6]] -> 1 + 6 = 7
    // [[3,4,7,8]] -> 3 + 8 = 11
    // [[9,10,13,14]] -> 9 + 14 = 23
    // [[11,12,15,16]] -> 11 + 16 = 27
    assert_eq!(result, vec![7.0, 11.0, 23.0, 27.0]);
}

#[test]
fn test_conv3d_simple_f32() {
    // 2x2x2 cube: [1..8]
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
    // 2x2x2 kernel (identity-like: only first element is 1)
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

// Helper functions for grad_weight tests

#[allow(clippy::too_many_arguments)]
fn run_conv1d_grad_weight<T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits>(
    input: &[T],
    grad_output: &[T],
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    input_length: usize,
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    kernel: hodu_cuda_kernels::kernels::Kernel,
) -> Vec<T> {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let grad_output_dev = stream.memcpy_stod(grad_output).unwrap();

    let grad_weight_size = out_channels * in_channels * kernel_size;
    let mut grad_weight: cudarc::driver::CudaSlice<T> = unsafe { stream.alloc(grad_weight_size).unwrap() };
    stream.memset_zeros(&mut grad_weight).unwrap();

    // Generic metadata layout for conv1d_grad_weight (input_ndim=3, spatial_dims=1):
    // [num_els, input_ndim, spatial_dims,
    //  input_shape(3), grad_output_shape(3), weight_shape(3),
    //  input_strides(3), grad_output_strides(3),
    //  input_offset, grad_output_offset,
    //  stride, padding, dilation]
    let metadata = vec![
        grad_weight_size, // num_els
        3,                // input_ndim
        1,                // spatial_dims
        // input_shape: [batch, in_channels, in_width]
        batch,
        in_channels,
        input_length,
        // grad_output_shape: [batch, out_channels, out_width]
        batch,
        out_channels,
        output_length,
        // weight_shape: [out_channels, in_channels, kernel_width]
        out_channels,
        in_channels,
        kernel_size,
        // input_strides: [in_channels * in_width, in_width, 1]
        in_channels * input_length,
        input_length,
        1,
        // grad_output_strides: [out_channels * out_width, out_width, 1]
        out_channels * output_length,
        output_length,
        1,
        // input_offset, grad_output_offset
        0,
        0,
        // stride, padding, dilation
        stride,
        padding,
        1,
    ];

    call_ops_conv_grad_weight(
        kernel,
        &kernels,
        &device,
        &input_dev,
        &grad_output_dev,
        &mut grad_weight,
        &metadata,
    )
    .unwrap();

    let mut results = vec![unsafe { core::mem::zeroed() }; grad_weight_size];
    stream.memcpy_dtoh(&grad_weight, &mut results).unwrap();
    results
}

#[allow(clippy::too_many_arguments)]
fn run_conv2d_grad_weight<T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits>(
    input: &[T],
    grad_output: &[T],
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    kernel: hodu_cuda_kernels::kernels::Kernel,
) -> Vec<T> {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let grad_output_dev = stream.memcpy_stod(grad_output).unwrap();

    let grad_weight_size = out_channels * in_channels * kernel_h * kernel_w;
    let mut grad_weight: cudarc::driver::CudaSlice<T> = unsafe { stream.alloc(grad_weight_size).unwrap() };
    stream.memset_zeros(&mut grad_weight).unwrap();

    // Generic metadata layout for conv2d_grad_weight (input_ndim=4, spatial_dims=2):
    let metadata = vec![
        grad_weight_size, // num_els
        4,                // input_ndim
        2,                // spatial_dims
        // input_shape: [batch, in_channels, in_height, in_width]
        batch,
        in_channels,
        input_height,
        input_width,
        // grad_output_shape: [batch, out_channels, out_height, out_width]
        batch,
        out_channels,
        output_height,
        output_width,
        // weight_shape: [out_channels, in_channels, kernel_height, kernel_width]
        out_channels,
        in_channels,
        kernel_h,
        kernel_w,
        // input_strides: [C*H*W, H*W, W, 1]
        in_channels * input_height * input_width,
        input_height * input_width,
        input_width,
        1,
        // grad_output_strides: [C*H*W, H*W, W, 1]
        out_channels * output_height * output_width,
        output_height * output_width,
        output_width,
        1,
        // input_offset, grad_output_offset
        0,
        0,
        // stride_h, stride_w
        stride_h,
        stride_w,
        // padding_h, padding_w
        padding_h,
        padding_w,
        // dilation_h, dilation_w
        1,
        1,
    ];

    call_ops_conv_grad_weight(
        kernel,
        &kernels,
        &device,
        &input_dev,
        &grad_output_dev,
        &mut grad_weight,
        &metadata,
    )
    .unwrap();

    let mut results = vec![unsafe { core::mem::zeroed() }; grad_weight_size];
    stream.memcpy_dtoh(&grad_weight, &mut results).unwrap();
    results
}

#[allow(clippy::too_many_arguments)]
fn run_conv3d_grad_weight<T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits>(
    input: &[T],
    grad_output: &[T],
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    input_depth: usize,
    input_height: usize,
    input_width: usize,
    output_depth: usize,
    output_height: usize,
    output_width: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    padding_d: usize,
    padding_h: usize,
    padding_w: usize,
    kernel: hodu_cuda_kernels::kernels::Kernel,
) -> Vec<T> {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let grad_output_dev = stream.memcpy_stod(grad_output).unwrap();

    let grad_weight_size = out_channels * in_channels * kernel_d * kernel_h * kernel_w;
    let mut grad_weight: cudarc::driver::CudaSlice<T> = unsafe { stream.alloc(grad_weight_size).unwrap() };
    stream.memset_zeros(&mut grad_weight).unwrap();

    // Generic metadata layout for conv3d_grad_weight (input_ndim=5, spatial_dims=3):
    let metadata = vec![
        grad_weight_size, // num_els
        5,                // input_ndim
        3,                // spatial_dims
        // input_shape: [batch, in_channels, in_depth, in_height, in_width]
        batch,
        in_channels,
        input_depth,
        input_height,
        input_width,
        // grad_output_shape: [batch, out_channels, out_depth, out_height, out_width]
        batch,
        out_channels,
        output_depth,
        output_height,
        output_width,
        // weight_shape: [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]
        out_channels,
        in_channels,
        kernel_d,
        kernel_h,
        kernel_w,
        // input_strides: [C*D*H*W, D*H*W, H*W, W, 1]
        in_channels * input_depth * input_height * input_width,
        input_depth * input_height * input_width,
        input_height * input_width,
        input_width,
        1,
        // grad_output_strides: [C*D*H*W, D*H*W, H*W, W, 1]
        out_channels * output_depth * output_height * output_width,
        output_depth * output_height * output_width,
        output_height * output_width,
        output_width,
        1,
        // input_offset, grad_output_offset
        0,
        0,
        // stride_d, stride_h, stride_w
        stride_d,
        stride_h,
        stride_w,
        // padding_d, padding_h, padding_w
        padding_d,
        padding_h,
        padding_w,
        // dilation_d, dilation_h, dilation_w
        1,
        1,
        1,
    ];

    call_ops_conv_grad_weight(
        kernel,
        &kernels,
        &device,
        &input_dev,
        &grad_output_dev,
        &mut grad_weight,
        &metadata,
    )
    .unwrap();

    let mut results = vec![unsafe { core::mem::zeroed() }; grad_weight_size];
    stream.memcpy_dtoh(&grad_weight, &mut results).unwrap();
    results
}

// Grad weight tests

#[test]
fn test_conv1d_grad_weight_simple_f32() {
    // Input: [1, 2, 3, 4, 5]
    // grad_output from forward with weight [1, 0, -1]: [-2, -2, -2]
    // We use all ones grad_output for simplicity
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let grad_output: Vec<f32> = vec![1.0, 1.0, 1.0];

    let result: Vec<f32> = run_conv1d_grad_weight(
        &input,
        &grad_output,
        1, // batch
        1, // in_channels
        1, // out_channels
        5, // input_length
        3, // output_length
        3, // kernel_size
        1, // stride
        0, // padding
        conv1d_grad_weight::F32,
    );

    assert_eq!(result.len(), 3);
    // grad_weight[0] = 1*1 + 2*1 + 3*1 = 6
    // grad_weight[1] = 2*1 + 3*1 + 4*1 = 9
    // grad_weight[2] = 3*1 + 4*1 + 5*1 = 12
    assert_eq!(result, vec![6.0, 9.0, 12.0]);
}

#[test]
fn test_conv2d_grad_weight_simple_f32() {
    // Input: 3x3 [[1,2,3], [4,5,6], [7,8,9]]
    // grad_output: 2x2 all ones
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let grad_output: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

    let result: Vec<f32> = run_conv2d_grad_weight(
        &input,
        &grad_output,
        1, // batch
        1, // in_channels
        1, // out_channels
        3, // input_height
        3, // input_width
        2, // output_height
        2, // output_width
        2, // kernel_h
        2, // kernel_w
        1, // stride_h
        1, // stride_w
        0, // padding_h
        0, // padding_w
        conv2d_grad_weight::F32,
    );

    assert_eq!(result.len(), 4);
    // grad_weight = sum of 4 patches:
    // patch1: [[1,2],[4,5]], patch2: [[2,3],[5,6]]
    // patch3: [[4,5],[7,8]], patch4: [[5,6],[8,9]]
    // grad_weight[0,0] = 1+2+4+5 = 12
    // grad_weight[0,1] = 2+3+5+6 = 16
    // grad_weight[1,0] = 4+5+7+8 = 24
    // grad_weight[1,1] = 5+6+8+9 = 28
    assert_eq!(result, vec![12.0, 16.0, 24.0, 28.0]);
}

#[test]
fn test_conv3d_grad_weight_simple_f32() {
    // Input: 2x2x2 volume
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let grad_output: Vec<f32> = vec![1.0];

    let result: Vec<f32> = run_conv3d_grad_weight(
        &input,
        &grad_output,
        1, // batch
        1, // in_channels
        1, // out_channels
        2, // input_depth
        2, // input_height
        2, // input_width
        1, // output_depth
        1, // output_height
        1, // output_width
        2, // kernel_d
        2, // kernel_h
        2, // kernel_w
        1, // stride_d
        1, // stride_h
        1, // stride_w
        0, // padding_d
        0, // padding_h
        0, // padding_w
        conv3d_grad_weight::F32,
    );

    assert_eq!(result.len(), 8);
    // grad_weight = entire input volume (one patch)
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_conv1d_grad_weight_multi_channel_f32() {
    // Input: 2 channels, 4 elements each
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // grad_output: 1 output channel, 3 output positions
    let grad_output: Vec<f32> = vec![1.0, 1.0, 1.0];

    let result: Vec<f32> = run_conv1d_grad_weight(
        &input,
        &grad_output,
        1, // batch
        2, // in_channels
        1, // out_channels
        4, // input_length
        3, // output_length
        2, // kernel_size
        1, // stride
        0, // padding
        conv1d_grad_weight::F32,
    );

    assert_eq!(result.len(), 4); // out_channels(1) * in_channels(2) * kernel_size(2)
                                 // For channel 0: grad[0] = 1+2+3 = 6, grad[1] = 2+3+4 = 9
                                 // For channel 1: grad[2] = 5+6+7 = 18, grad[3] = 6+7+8 = 21
    assert_eq!(result, vec![6.0, 9.0, 18.0, 21.0]);
}

#[test]
fn test_conv2d_grad_weight_stride_f32() {
    // Input: 4x4
    let input: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    // grad_output: 2x2 with stride=2
    let grad_output: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

    let result: Vec<f32> = run_conv2d_grad_weight(
        &input,
        &grad_output,
        1, // batch
        1, // in_channels
        1, // out_channels
        4, // input_height
        4, // input_width
        2, // output_height
        2, // output_width
        2, // kernel_h
        2, // kernel_w
        2, // stride_h
        2, // stride_w
        0, // padding_h
        0, // padding_w
        conv2d_grad_weight::F32,
    );

    assert_eq!(result.len(), 4);
    // 4 patches with stride 2:
    // patch1 [0,0]: [[1,2],[5,6]], patch2 [0,2]: [[3,4],[7,8]]
    // patch3 [2,0]: [[9,10],[13,14]], patch4 [2,2]: [[11,12],[15,16]]
    // grad[0,0] = 1+3+9+11 = 24
    // grad[0,1] = 2+4+10+12 = 28
    // grad[1,0] = 5+7+13+15 = 40
    // grad[1,1] = 6+8+14+16 = 44
    assert_eq!(result, vec![24.0, 28.0, 40.0, 44.0]);
}

#[test]
fn test_conv3d_grad_weight_multi_channel_f32() {
    // Input: 2 in_channels, 2x2x2 each
    let input: Vec<f32> = vec![
        // channel 0
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // channel 1
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    // grad_output: 1 out_channel, 1x1x1
    let grad_output: Vec<f32> = vec![1.0];

    let result: Vec<f32> = run_conv3d_grad_weight(
        &input,
        &grad_output,
        1, // batch
        2, // in_channels
        1, // out_channels
        2, // input_depth
        2, // input_height
        2, // input_width
        1, // output_depth
        1, // output_height
        1, // output_width
        2, // kernel_d
        2, // kernel_h
        2, // kernel_w
        1, // stride_d
        1, // stride_h
        1, // stride_w
        0, // padding_d
        0, // padding_h
        0, // padding_w
        conv3d_grad_weight::F32,
    );

    assert_eq!(result.len(), 16); // 1 * 2 * 2 * 2 * 2
                                  // First 8 elements from channel 0, next 8 from channel 1
    assert_eq!(result[..8], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(result[8..], vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
}

#[test]
fn test_conv2d_grad_weight_batch_f32() {
    // Batch of 2, 2x2 images
    let input: Vec<f32> = vec![
        // batch 0
        1.0, 2.0, 3.0, 4.0, // batch 1
        5.0, 6.0, 7.0, 8.0,
    ];
    // grad_output: batch of 2, 1x1 each
    let grad_output: Vec<f32> = vec![1.0, 1.0];

    let result: Vec<f32> = run_conv2d_grad_weight(
        &input,
        &grad_output,
        2, // batch
        1, // in_channels
        1, // out_channels
        2, // input_height
        2, // input_width
        1, // output_height
        1, // output_width
        2, // kernel_h
        2, // kernel_w
        1, // stride_h
        1, // stride_w
        0, // padding_h
        0, // padding_w
        conv2d_grad_weight::F32,
    );

    assert_eq!(result.len(), 4);
    // Sum over batch:
    // batch 0: [1,2,3,4], batch 1: [5,6,7,8]
    // grad = [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
}
