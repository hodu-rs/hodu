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

    call_conv(
        conv1d::F32,
        input.as_ptr() as *const std::ffi::c_void,
        weight.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
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

    call_conv(
        conv1d::F32,
        input.as_ptr() as *const std::ffi::c_void,
        weight.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
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

    call_conv(
        conv2d::F32,
        input.as_ptr() as *const std::ffi::c_void,
        weight.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
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

    call_conv(
        conv2d::F32,
        input.as_ptr() as *const std::ffi::c_void,
        weight.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
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

    call_conv(
        conv3d::F32,
        input.as_ptr() as *const std::ffi::c_void,
        weight.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
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

    call_conv(
        conv_transpose1d::F32,
        input.as_ptr() as *const std::ffi::c_void,
        weight.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
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

    call_conv(
        conv_transpose2d::F32,
        input.as_ptr() as *const std::ffi::c_void,
        weight.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
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

    call_conv(
        conv_transpose3d::F32,
        input.as_ptr() as *const std::ffi::c_void,
        weight.as_ptr() as *const std::ffi::c_void,
        output.as_mut_ptr() as *mut std::ffi::c_void,
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

    let metadata = vec![
        out_channels * in_channels * kernel_width,
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

    call_conv_grad_weight(
        conv1d_grad_weight::F32,
        input.as_ptr() as *const std::ffi::c_void,
        grad_output.as_ptr() as *const std::ffi::c_void,
        grad_weight.as_mut_ptr() as *mut std::ffi::c_void,
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

    let metadata = vec![
        out_channels * in_channels * kernel_height * kernel_width,
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

    call_conv_grad_weight(
        conv2d_grad_weight::F32,
        input.as_ptr() as *const std::ffi::c_void,
        grad_output.as_ptr() as *const std::ffi::c_void,
        grad_weight.as_mut_ptr() as *mut std::ffi::c_void,
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

    let metadata = vec![
        out_channels * in_channels * kernel_depth * kernel_height * kernel_width,
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

    call_conv_grad_weight(
        conv3d_grad_weight::F32,
        input.as_ptr() as *const std::ffi::c_void,
        grad_output.as_ptr() as *const std::ffi::c_void,
        grad_weight.as_mut_ptr() as *mut std::ffi::c_void,
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

    let metadata = vec![
        in_channels * out_channels * kernel_width,
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

    call_conv_grad_weight(
        conv_transpose1d_grad_weight::F32,
        input.as_ptr() as *const std::ffi::c_void,
        grad_output.as_ptr() as *const std::ffi::c_void,
        grad_weight.as_mut_ptr() as *mut std::ffi::c_void,
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

    let metadata = vec![
        in_channels * out_channels * kernel_height * kernel_width,
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

    call_conv_grad_weight(
        conv_transpose2d_grad_weight::F32,
        input.as_ptr() as *const std::ffi::c_void,
        grad_output.as_ptr() as *const std::ffi::c_void,
        grad_weight.as_mut_ptr() as *mut std::ffi::c_void,
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

    let metadata = vec![
        in_channels * out_channels * kernel_depth * kernel_height * kernel_width,
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

    call_conv_grad_weight(
        conv_transpose3d_grad_weight::F32,
        input.as_ptr() as *const std::ffi::c_void,
        grad_output.as_ptr() as *const std::ffi::c_void,
        grad_weight.as_mut_ptr() as *mut std::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(grad_weight.iter().sum::<f32>(), 288.0);
}
