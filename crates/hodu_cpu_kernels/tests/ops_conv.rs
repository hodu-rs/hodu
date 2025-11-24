use hodu_cpu_kernels::*;

fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t * b) / b).collect()
}

#[test]
fn test_conv1d_f32() {
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_width = 5;
    let kernel_width = 3;
    let stride = 1;
    let padding = 0;
    let dilation = 1;
    let out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let weight = [1.0f32, 0.0, -1.0];
    let mut output = vec![0.0f32; batch * out_channels * out_width];

    let metadata = vec![
        batch * out_channels * out_width,
        batch,
        in_channels,
        out_channels,
        in_width,
        kernel_width,
        out_width,
        stride,
        padding,
        dilation,
        0,
        0,
    ];

    call_ops_conv(
        conv1d::F32,
        input.as_ptr() as *const core::ffi::c_void,
        weight.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output, 4), vec![-2.0, -2.0, -2.0]);
}

#[test]
fn test_conv1d_f32_stride() {
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_width = 7;
    let kernel_width = 3;
    let stride = 2;
    let padding = 1;
    let dilation = 1;
    let out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let weight = [1.0f32, 2.0, 1.0];
    let mut output = vec![0.0f32; batch * out_channels * out_width];

    let metadata = vec![
        batch * out_channels * out_width,
        batch,
        in_channels,
        out_channels,
        in_width,
        kernel_width,
        out_width,
        stride,
        padding,
        dilation,
        0,
        0,
    ];

    call_ops_conv(
        conv1d::F32,
        input.as_ptr() as *const core::ffi::c_void,
        weight.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output, 4), vec![4.0, 12.0, 20.0, 20.0]);
}

#[test]
fn test_conv2d_f32() {
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_height = 3;
    let in_width = 3;
    let kernel_height = 2;
    let kernel_width = 2;
    let stride_h = 1;
    let stride_w = 1;
    let padding_h = 0;
    let padding_w = 0;
    let dilation_h = 1;
    let dilation_w = 1;
    let out_height = (in_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    let out_width = (in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let weight = [1.0f32, 0.0, 0.0, 1.0];
    let mut output = vec![0.0f32; batch * out_channels * out_height * out_width];

    let metadata = vec![
        batch * out_channels * out_height * out_width,
        batch,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        0,
        0,
    ];

    call_ops_conv(
        conv2d::F32,
        input.as_ptr() as *const core::ffi::c_void,
        weight.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output, 4), vec![6.0, 8.0, 12.0, 14.0]);
}

#[test]
fn test_conv2d_f32_with_padding() {
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_height = 3;
    let in_width = 3;
    let kernel_height = 3;
    let kernel_width = 3;
    let stride_h = 1;
    let stride_w = 1;
    let padding_h = 1;
    let padding_w = 1;
    let dilation_h = 1;
    let dilation_w = 1;
    let out_height = (in_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    let out_width = (in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let weight = [0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let mut output = vec![0.0f32; batch * out_channels * out_height * out_width];

    let metadata = vec![
        batch * out_channels * out_height * out_width,
        batch,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        0,
        0,
    ];

    call_ops_conv(
        conv2d::F32,
        input.as_ptr() as *const core::ffi::c_void,
        weight.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn test_conv3d_f32() {
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_depth = 3;
    let in_height = 3;
    let in_width = 3;
    let kernel_depth = 2;
    let kernel_height = 2;
    let kernel_width = 2;
    let stride_d = 1;
    let stride_h = 1;
    let stride_w = 1;
    let padding_d = 0;
    let padding_h = 0;
    let padding_w = 0;
    let dilation_d = 1;
    let dilation_h = 1;
    let dilation_w = 1;
    let out_depth = (in_depth + 2 * padding_d - dilation_d * (kernel_depth - 1) - 1) / stride_d + 1;
    let out_height = (in_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    let out_width = (in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;

    let input = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
        20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
    ];
    let weight = [1.0f32; 8];
    let mut output = vec![0.0f32; batch * out_channels * out_depth * out_height * out_width];

    let metadata = vec![
        batch * out_channels * out_depth * out_height * out_width,
        batch,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        out_depth,
        out_height,
        out_width,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
        0,
        0,
    ];

    call_ops_conv(
        conv3d::F32,
        input.as_ptr() as *const core::ffi::c_void,
        weight.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(
        approx(output, 4),
        vec![60.0, 68.0, 84.0, 92.0, 132.0, 140.0, 156.0, 164.0]
    );
}

#[test]
fn test_conv_transpose1d_f32() {
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_width = 3;
    let kernel_width = 2;
    let stride = 2;
    let padding = 0;
    let dilation = 1;
    let out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_width - 1) + 1;

    let input = [1.0f32, 2.0, 3.0];
    let weight = [1.0f32, 2.0];
    let mut output = vec![0.0f32; batch * out_channels * out_width];

    let metadata = vec![
        batch * out_channels * out_width,
        batch,
        in_channels,
        out_channels,
        in_width,
        kernel_width,
        out_width,
        stride,
        padding,
        dilation,
        0,
        0,
    ];

    call_ops_conv(
        conv_transpose1d::F32,
        input.as_ptr() as *const core::ffi::c_void,
        weight.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(output, 4), vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);
}

#[test]
fn test_conv_transpose2d_f32() {
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_height = 2;
    let in_width = 2;
    let kernel_height = 2;
    let kernel_width = 2;
    let stride_h = 2;
    let stride_w = 2;
    let padding_h = 0;
    let padding_w = 0;
    let dilation_h = 1;
    let dilation_w = 1;
    let out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_height - 1) + 1;
    let out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_width - 1) + 1;

    let input = [1.0f32, 2.0, 3.0, 4.0];
    let weight = [1.0f32, 0.0, 0.0, 0.0];
    let mut output = vec![0.0f32; batch * out_channels * out_height * out_width];

    let metadata = vec![
        batch * out_channels * out_height * out_width,
        batch,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        0,
        0,
    ];

    call_ops_conv(
        conv_transpose2d::F32,
        input.as_ptr() as *const core::ffi::c_void,
        weight.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(
        approx(output, 4),
        vec![1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn test_conv_transpose3d_f32() {
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_depth = 2;
    let in_height = 2;
    let in_width = 2;
    let kernel_depth = 2;
    let kernel_height = 2;
    let kernel_width = 2;
    let stride_d = 2;
    let stride_h = 2;
    let stride_w = 2;
    let padding_d = 0;
    let padding_h = 0;
    let padding_w = 0;
    let dilation_d = 1;
    let dilation_h = 1;
    let dilation_w = 1;
    let out_depth = (in_depth - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_depth - 1) + 1;
    let out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_height - 1) + 1;
    let out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_width - 1) + 1;

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = [1.0f32; 8];
    let mut output = vec![0.0f32; batch * out_channels * out_depth * out_height * out_width];

    let metadata = vec![
        batch * out_channels * out_depth * out_height * out_width,
        batch,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        out_depth,
        out_height,
        out_width,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
        0,
        0,
    ];

    call_ops_conv(
        conv_transpose3d::F32,
        input.as_ptr() as *const core::ffi::c_void,
        weight.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(output.iter().sum::<f32>(), 288.0);
}

#[test]
fn test_conv1d_grad_weight_f32() {
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_width = 5;
    let kernel_width = 3;
    let stride = 1;
    let padding = 0;
    let dilation = 1;
    let out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let grad_output = [1.0f32, 1.0, 1.0];
    let mut grad_weight = vec![0.0f32; out_channels * in_channels * kernel_width];

    // Generic metadata layout for conv1d_grad_weight:
    // metadata[0]: num_els
    // metadata[1]: input_ndim (3 for conv1d)
    // metadata[2]: spatial_dims (1 for conv1d)
    // metadata[3..6]: input_shape [batch, in_channels, in_width]
    // metadata[6..9]: grad_output_shape [batch, out_channels, out_width]
    // metadata[9..12]: weight_shape [out_channels, in_channels, kernel_width]
    // metadata[12..15]: input_strides
    // metadata[15..18]: grad_output_strides
    // metadata[18]: input_offset
    // metadata[19]: grad_output_offset
    // metadata[20]: stride
    // metadata[21]: padding
    // metadata[22]: dilation
    let input_ndim = 3;
    let spatial_dims = 1;
    let input_strides = [in_channels * in_width, in_width, 1];
    let grad_output_strides = [out_channels * out_width, out_width, 1];
    let metadata = vec![
        out_channels * in_channels * kernel_width, // num_els
        input_ndim,
        spatial_dims,
        batch,
        in_channels,
        in_width, // input_shape
        batch,
        out_channels,
        out_width, // grad_output_shape
        out_channels,
        in_channels,
        kernel_width, // weight_shape
        input_strides[0],
        input_strides[1],
        input_strides[2], // input_strides
        grad_output_strides[0],
        grad_output_strides[1],
        grad_output_strides[2], // grad_output_strides
        0,                      // input_offset
        0,                      // grad_output_offset
        stride,                 // stride
        padding,                // padding
        dilation,               // dilation
    ];

    call_ops_conv_grad_weight(
        conv1d_grad_weight::F32,
        input.as_ptr() as *const core::ffi::c_void,
        grad_output.as_ptr() as *const core::ffi::c_void,
        grad_weight.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(grad_weight, 4), vec![6.0, 9.0, 12.0]);
}

#[test]
fn test_conv2d_grad_weight_f32() {
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_height = 3;
    let in_width = 3;
    let kernel_height = 2;
    let kernel_width = 2;
    let stride_h = 1;
    let stride_w = 1;
    let padding_h = 0;
    let padding_w = 0;
    let dilation_h = 1;
    let dilation_w = 1;
    let out_height = (in_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    let out_width = (in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let grad_output = [1.0f32, 1.0, 1.0, 1.0];
    let mut grad_weight = vec![0.0f32; out_channels * in_channels * kernel_height * kernel_width];

    // Generic metadata layout for conv2d_grad_weight:
    // metadata[0]: num_els
    // metadata[1]: input_ndim (4 for conv2d)
    // metadata[2]: spatial_dims (2 for conv2d)
    // metadata[3..7]: input_shape [batch, in_channels, in_height, in_width]
    // metadata[7..11]: grad_output_shape [batch, out_channels, out_height, out_width]
    // metadata[11..15]: weight_shape [out_channels, in_channels, kernel_height, kernel_width]
    // metadata[15..19]: input_strides
    // metadata[19..23]: grad_output_strides
    // metadata[23]: input_offset
    // metadata[24]: grad_output_offset
    // metadata[25..27]: stride [stride_h, stride_w]
    // metadata[27..29]: padding [padding_h, padding_w]
    // metadata[29..31]: dilation [dilation_h, dilation_w]
    let input_ndim = 4;
    let spatial_dims = 2;
    let input_strides = [in_channels * in_height * in_width, in_height * in_width, in_width, 1];
    let grad_output_strides = [
        out_channels * out_height * out_width,
        out_height * out_width,
        out_width,
        1,
    ];
    let metadata = vec![
        out_channels * in_channels * kernel_height * kernel_width, // num_els
        input_ndim,
        spatial_dims,
        batch,
        in_channels,
        in_height,
        in_width, // input_shape
        batch,
        out_channels,
        out_height,
        out_width, // grad_output_shape
        out_channels,
        in_channels,
        kernel_height,
        kernel_width, // weight_shape
        input_strides[0],
        input_strides[1],
        input_strides[2],
        input_strides[3], // input_strides
        grad_output_strides[0],
        grad_output_strides[1],
        grad_output_strides[2],
        grad_output_strides[3], // grad_output_strides
        0,                      // input_offset
        0,                      // grad_output_offset
        stride_h,
        stride_w, // stride
        padding_h,
        padding_w, // padding
        dilation_h,
        dilation_w, // dilation
    ];

    call_ops_conv_grad_weight(
        conv2d_grad_weight::F32,
        input.as_ptr() as *const core::ffi::c_void,
        grad_output.as_ptr() as *const core::ffi::c_void,
        grad_weight.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(grad_weight, 4), vec![12.0, 16.0, 24.0, 28.0]);
}

#[test]
fn test_conv3d_grad_weight_f32() {
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_depth = 2;
    let in_height = 2;
    let in_width = 2;
    let kernel_depth = 2;
    let kernel_height = 2;
    let kernel_width = 2;
    let stride_d = 1;
    let stride_h = 1;
    let stride_w = 1;
    let padding_d = 0;
    let padding_h = 0;
    let padding_w = 0;
    let dilation_d = 1;
    let dilation_h = 1;
    let dilation_w = 1;
    let out_depth = (in_depth + 2 * padding_d - dilation_d * (kernel_depth - 1) - 1) / stride_d + 1;
    let out_height = (in_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    let out_width = (in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let grad_output = [1.0f32];
    let mut grad_weight = vec![0.0f32; out_channels * in_channels * kernel_depth * kernel_height * kernel_width];

    // Generic metadata layout for conv3d_grad_weight:
    // metadata[0]: num_els
    // metadata[1]: input_ndim (5 for conv3d)
    // metadata[2]: spatial_dims (3 for conv3d)
    // metadata[3..8]: input_shape [batch, in_channels, in_depth, in_height, in_width]
    // metadata[8..13]: grad_output_shape [batch, out_channels, out_depth, out_height, out_width]
    // metadata[13..18]: weight_shape [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]
    // metadata[18..23]: input_strides
    // metadata[23..28]: grad_output_strides
    // metadata[28]: input_offset
    // metadata[29]: grad_output_offset
    // metadata[30..33]: stride [stride_d, stride_h, stride_w]
    // metadata[33..36]: padding [padding_d, padding_h, padding_w]
    // metadata[36..39]: dilation [dilation_d, dilation_h, dilation_w]
    let input_ndim = 5;
    let spatial_dims = 3;
    let input_strides = [
        in_channels * in_depth * in_height * in_width,
        in_depth * in_height * in_width,
        in_height * in_width,
        in_width,
        1,
    ];
    let grad_output_strides = [
        out_channels * out_depth * out_height * out_width,
        out_depth * out_height * out_width,
        out_height * out_width,
        out_width,
        1,
    ];
    let metadata = vec![
        out_channels * in_channels * kernel_depth * kernel_height * kernel_width, // num_els
        input_ndim,
        spatial_dims,
        batch,
        in_channels,
        in_depth,
        in_height,
        in_width, // input_shape
        batch,
        out_channels,
        out_depth,
        out_height,
        out_width, // grad_output_shape
        out_channels,
        in_channels,
        kernel_depth,
        kernel_height,
        kernel_width, // weight_shape
        input_strides[0],
        input_strides[1],
        input_strides[2],
        input_strides[3],
        input_strides[4], // input_strides
        grad_output_strides[0],
        grad_output_strides[1],
        grad_output_strides[2],
        grad_output_strides[3],
        grad_output_strides[4], // grad_output_strides
        0,                      // input_offset
        0,                      // grad_output_offset
        stride_d,
        stride_h,
        stride_w, // stride
        padding_d,
        padding_h,
        padding_w, // padding
        dilation_d,
        dilation_h,
        dilation_w, // dilation
    ];

    call_ops_conv_grad_weight(
        conv3d_grad_weight::F32,
        input.as_ptr() as *const core::ffi::c_void,
        grad_output.as_ptr() as *const core::ffi::c_void,
        grad_weight.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(grad_weight, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_conv_transpose1d_grad_weight_f32() {
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_width = 3;
    let kernel_width = 2;
    let stride = 2;
    let padding = 0;
    let dilation = 1;
    let out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_width - 1) + 1;

    let input = [1.0f32, 2.0, 3.0];
    let grad_output = vec![1.0f32; out_width];
    let mut grad_weight = vec![0.0f32; in_channels * out_channels * kernel_width];

    // Generic metadata layout for conv_transpose1d_grad_weight:
    // Same structure as conv1d_grad_weight but with transposed semantics
    let input_ndim = 3;
    let spatial_dims = 1;
    let input_strides = [in_channels * in_width, in_width, 1];
    let grad_output_strides = [out_channels * out_width, out_width, 1];
    let metadata = vec![
        in_channels * out_channels * kernel_width, // num_els
        input_ndim,
        spatial_dims,
        batch,
        in_channels,
        in_width, // input_shape
        batch,
        out_channels,
        out_width, // grad_output_shape
        in_channels,
        out_channels,
        kernel_width, // weight_shape
        input_strides[0],
        input_strides[1],
        input_strides[2], // input_strides
        grad_output_strides[0],
        grad_output_strides[1],
        grad_output_strides[2], // grad_output_strides
        0,                      // input_offset
        0,                      // grad_output_offset
        stride,                 // stride
        padding,                // padding
        dilation,               // dilation
    ];

    call_ops_conv_grad_weight(
        conv_transpose1d_grad_weight::F32,
        input.as_ptr() as *const core::ffi::c_void,
        grad_output.as_ptr() as *const core::ffi::c_void,
        grad_weight.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(grad_weight, 4), vec![6.0, 6.0]);
}

#[test]
fn test_conv_transpose2d_grad_weight_f32() {
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_height = 2;
    let in_width = 2;
    let kernel_height = 2;
    let kernel_width = 2;
    let stride_h = 2;
    let stride_w = 2;
    let padding_h = 0;
    let padding_w = 0;
    let dilation_h = 1;
    let dilation_w = 1;
    let out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_height - 1) + 1;
    let out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_width - 1) + 1;

    let input = [1.0f32, 2.0, 3.0, 4.0];
    let grad_output = vec![1.0f32; out_height * out_width];
    let mut grad_weight = vec![0.0f32; in_channels * out_channels * kernel_height * kernel_width];

    // Generic metadata layout for conv_transpose2d_grad_weight
    let input_ndim = 4;
    let spatial_dims = 2;
    let input_strides = [in_channels * in_height * in_width, in_height * in_width, in_width, 1];
    let grad_output_strides = [
        out_channels * out_height * out_width,
        out_height * out_width,
        out_width,
        1,
    ];
    let metadata = vec![
        in_channels * out_channels * kernel_height * kernel_width, // num_els
        input_ndim,
        spatial_dims,
        batch,
        in_channels,
        in_height,
        in_width, // input_shape
        batch,
        out_channels,
        out_height,
        out_width, // grad_output_shape
        in_channels,
        out_channels,
        kernel_height,
        kernel_width, // weight_shape
        input_strides[0],
        input_strides[1],
        input_strides[2],
        input_strides[3], // input_strides
        grad_output_strides[0],
        grad_output_strides[1],
        grad_output_strides[2],
        grad_output_strides[3], // grad_output_strides
        0,                      // input_offset
        0,                      // grad_output_offset
        stride_h,
        stride_w, // stride
        padding_h,
        padding_w, // padding
        dilation_h,
        dilation_w, // dilation
    ];

    call_ops_conv_grad_weight(
        conv_transpose2d_grad_weight::F32,
        input.as_ptr() as *const core::ffi::c_void,
        grad_output.as_ptr() as *const core::ffi::c_void,
        grad_weight.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(approx(grad_weight, 4), vec![10.0, 10.0, 10.0, 10.0]);
}

#[test]
fn test_conv_transpose3d_grad_weight_f32() {
    let batch = 1;
    let in_channels = 1;
    let out_channels = 1;
    let in_depth = 2;
    let in_height = 2;
    let in_width = 2;
    let kernel_depth = 2;
    let kernel_height = 2;
    let kernel_width = 2;
    let stride_d = 2;
    let stride_h = 2;
    let stride_w = 2;
    let padding_d = 0;
    let padding_h = 0;
    let padding_w = 0;
    let dilation_d = 1;
    let dilation_h = 1;
    let dilation_w = 1;
    let out_depth = (in_depth - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_depth - 1) + 1;
    let out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_height - 1) + 1;
    let out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_width - 1) + 1;

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let grad_output = vec![1.0f32; out_depth * out_height * out_width];
    let mut grad_weight = vec![0.0f32; in_channels * out_channels * kernel_depth * kernel_height * kernel_width];

    // Generic metadata layout for conv_transpose3d_grad_weight
    let input_ndim = 5;
    let spatial_dims = 3;
    let input_strides = [
        in_channels * in_depth * in_height * in_width,
        in_depth * in_height * in_width,
        in_height * in_width,
        in_width,
        1,
    ];
    let grad_output_strides = [
        out_channels * out_depth * out_height * out_width,
        out_depth * out_height * out_width,
        out_height * out_width,
        out_width,
        1,
    ];
    let metadata = vec![
        in_channels * out_channels * kernel_depth * kernel_height * kernel_width, // num_els
        input_ndim,
        spatial_dims,
        batch,
        in_channels,
        in_depth,
        in_height,
        in_width, // input_shape
        batch,
        out_channels,
        out_depth,
        out_height,
        out_width, // grad_output_shape
        in_channels,
        out_channels,
        kernel_depth,
        kernel_height,
        kernel_width, // weight_shape
        input_strides[0],
        input_strides[1],
        input_strides[2],
        input_strides[3],
        input_strides[4], // input_strides
        grad_output_strides[0],
        grad_output_strides[1],
        grad_output_strides[2],
        grad_output_strides[3],
        grad_output_strides[4], // grad_output_strides
        0,                      // input_offset
        0,                      // grad_output_offset
        stride_d,
        stride_h,
        stride_w, // stride
        padding_d,
        padding_h,
        padding_w, // padding
        dilation_d,
        dilation_h,
        dilation_w, // dilation
    ];

    call_ops_conv_grad_weight(
        conv_transpose3d_grad_weight::F32,
        input.as_ptr() as *const core::ffi::c_void,
        grad_output.as_ptr() as *const core::ffi::c_void,
        grad_weight.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(grad_weight.iter().sum::<f32>(), 288.0);
}
