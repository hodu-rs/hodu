use crate::{
    backends::op::conv::{
        ParamsConv1D, ParamsConv2D, ParamsConv3D, ParamsConvTranspose1D, ParamsConvTranspose2D, ParamsConvTranspose3D,
    },
    compat::*,
    error::HoduResult,
    types::layout::Layout,
};

// Conv1D
pub fn conv1d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    weight_storage: &[T],
    weight_layout: &Layout,
    params: &ParamsConv1D,
) -> HoduResult<Vec<T>> {
    // Input: [batch, in_channels, length]
    // Weight: [out_channels, in_channels, kernel_size]
    // Output: [batch, out_channels, output_length]

    let output_length =
        (params.length_input + 2 * params.padding - params.dilation * (params.kernel_size - 1) - 1) / params.stride + 1;

    let output_size = params.batch_size * params.channels_output * output_length;
    let mut result = vec![T::default(); output_size];

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    // Iterate over batch
    for b in 0..params.batch_size {
        // Iterate over output channels
        for oc in 0..params.channels_output {
            // Iterate over output positions
            for ol in 0..output_length {
                let mut sum = T::default();

                // Iterate over input channels
                for ic in 0..params.channels_input {
                    // Iterate over kernel
                    for k in 0..params.kernel_size {
                        // Calculate input position with padding and dilation
                        let il = ol * params.stride + k * params.dilation;

                        // Check if position is within padded input
                        if il < params.padding || il >= params.length_input + params.padding {
                            continue; // Zero padding
                        }

                        let il_actual = il - params.padding;

                        // Get input value
                        let input_idx =
                            input_offset + b * input_strides[0] + ic * input_strides[1] + il_actual * input_strides[2];
                        let input_val = input_storage[input_idx];

                        // Get weight value
                        let weight_idx =
                            weight_offset + oc * weight_strides[0] + ic * weight_strides[1] + k * weight_strides[2];
                        let weight_val = weight_storage[weight_idx];

                        sum = sum + input_val * weight_val;
                    }
                }

                // Store result
                let output_idx = b * params.channels_output * output_length + oc * output_length + ol;
                result[output_idx] = sum;
            }
        }
    }

    Ok(result)
}

// Conv2D
pub fn conv2d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    weight_storage: &[T],
    weight_layout: &Layout,
    params: &ParamsConv2D,
) -> HoduResult<Vec<T>> {
    // Input: [batch, in_channels, height, width]
    // Weight: [out_channels, in_channels, kernel_h, kernel_w]
    // Output: [batch, out_channels, out_height, out_width]

    let output_height = (params.input_height + 2 * params.padding - params.dilation * (params.kernel_height - 1) - 1)
        / params.stride
        + 1;
    let output_width =
        (params.input_width + 2 * params.padding - params.dilation * (params.kernel_width - 1) - 1) / params.stride + 1;

    let output_size = params.batch_size * params.channels_output * output_height * output_width;
    let mut result = vec![T::default(); output_size];

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    // Iterate over batch
    for b in 0..params.batch_size {
        // Iterate over output channels
        for oc in 0..params.channels_output {
            // Iterate over output height
            for oh in 0..output_height {
                // Iterate over output width
                for ow in 0..output_width {
                    let mut sum = T::default();

                    // Iterate over input channels
                    for ic in 0..params.channels_input {
                        // Iterate over kernel height
                        for kh in 0..params.kernel_height {
                            // Iterate over kernel width
                            for kw in 0..params.kernel_width {
                                // Calculate input position
                                let ih = oh * params.stride + kh * params.dilation;
                                let iw = ow * params.stride + kw * params.dilation;

                                // Check bounds with padding
                                if ih < params.padding
                                    || ih >= params.input_height + params.padding
                                    || iw < params.padding
                                    || iw >= params.input_width + params.padding
                                {
                                    continue; // Zero padding
                                }

                                let ih_actual = ih - params.padding;
                                let iw_actual = iw - params.padding;

                                // Get input value
                                let input_idx = input_offset
                                    + b * input_strides[0]
                                    + ic * input_strides[1]
                                    + ih_actual * input_strides[2]
                                    + iw_actual * input_strides[3];
                                let input_val = input_storage[input_idx];

                                // Get weight value
                                let weight_idx = weight_offset
                                    + oc * weight_strides[0]
                                    + ic * weight_strides[1]
                                    + kh * weight_strides[2]
                                    + kw * weight_strides[3];
                                let weight_val = weight_storage[weight_idx];

                                sum = sum + input_val * weight_val;
                            }
                        }
                    }

                    // Store result
                    let output_idx = b * params.channels_output * output_height * output_width
                        + oc * output_height * output_width
                        + oh * output_width
                        + ow;
                    result[output_idx] = sum;
                }
            }
        }
    }

    Ok(result)
}

// Conv3D
pub fn conv3d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    weight_storage: &[T],
    weight_layout: &Layout,
    params: &ParamsConv3D,
) -> HoduResult<Vec<T>> {
    // Input: [batch, in_channels, depth, height, width]
    // Weight: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
    // Output: [batch, out_channels, out_depth, out_height, out_width]

    let output_depth =
        (params.input_depth + 2 * params.padding - params.dilation * (params.kernel_depth - 1) - 1) / params.stride + 1;
    let output_height = (params.input_height + 2 * params.padding - params.dilation * (params.kernel_height - 1) - 1)
        / params.stride
        + 1;
    let output_width =
        (params.input_width + 2 * params.padding - params.dilation * (params.kernel_width - 1) - 1) / params.stride + 1;

    let output_size = params.batch_size * params.channels_output * output_depth * output_height * output_width;
    let mut result = vec![T::default(); output_size];

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    // Iterate over batch
    for b in 0..params.batch_size {
        // Iterate over output channels
        for oc in 0..params.channels_output {
            // Iterate over output depth
            for od in 0..output_depth {
                // Iterate over output height
                for oh in 0..output_height {
                    // Iterate over output width
                    for ow in 0..output_width {
                        let mut sum = T::default();

                        // Iterate over input channels
                        for ic in 0..params.channels_input {
                            // Iterate over kernel depth
                            for kd in 0..params.kernel_depth {
                                // Iterate over kernel height
                                for kh in 0..params.kernel_height {
                                    // Iterate over kernel width
                                    for kw in 0..params.kernel_width {
                                        // Calculate input position
                                        let id = od * params.stride + kd * params.dilation;
                                        let ih = oh * params.stride + kh * params.dilation;
                                        let iw = ow * params.stride + kw * params.dilation;

                                        // Check bounds with padding
                                        if id < params.padding
                                            || id >= params.input_depth + params.padding
                                            || ih < params.padding
                                            || ih >= params.input_height + params.padding
                                            || iw < params.padding
                                            || iw >= params.input_width + params.padding
                                        {
                                            continue; // Zero padding
                                        }

                                        let id_actual = id - params.padding;
                                        let ih_actual = ih - params.padding;
                                        let iw_actual = iw - params.padding;

                                        // Get input value
                                        let input_idx = input_offset
                                            + b * input_strides[0]
                                            + ic * input_strides[1]
                                            + id_actual * input_strides[2]
                                            + ih_actual * input_strides[3]
                                            + iw_actual * input_strides[4];
                                        let input_val = input_storage[input_idx];

                                        // Get weight value
                                        let weight_idx = weight_offset
                                            + oc * weight_strides[0]
                                            + ic * weight_strides[1]
                                            + kd * weight_strides[2]
                                            + kh * weight_strides[3]
                                            + kw * weight_strides[4];
                                        let weight_val = weight_storage[weight_idx];

                                        sum = sum + input_val * weight_val;
                                    }
                                }
                            }
                        }

                        // Store result
                        let output_idx = b * params.channels_output * output_depth * output_height * output_width
                            + oc * output_depth * output_height * output_width
                            + od * output_height * output_width
                            + oh * output_width
                            + ow;
                        result[output_idx] = sum;
                    }
                }
            }
        }
    }

    Ok(result)
}

// ConvTranspose1D
pub fn conv_transpose1d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    weight_storage: &[T],
    weight_layout: &Layout,
    params: &ParamsConvTranspose1D,
) -> HoduResult<Vec<T>> {
    // Input: [batch, in_channels, length]
    // Weight: [in_channels, out_channels, kernel_size] (note: different from conv!)
    // Output: [batch, out_channels, output_length]

    let output_length = (params.length_input - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_size - 1)
        + params.output_padding
        + 1;

    let output_size = params.batch_size * params.channels_output * output_length;
    let mut result = vec![T::default(); output_size];

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    // Iterate over batch
    for b in 0..params.batch_size {
        // Iterate over input channels
        for ic in 0..params.channels_input {
            // Iterate over input positions
            for il in 0..params.length_input {
                // Get input value
                let input_idx = input_offset + b * input_strides[0] + ic * input_strides[1] + il * input_strides[2];
                let input_val = input_storage[input_idx];

                // Iterate over output channels
                for oc in 0..params.channels_output {
                    // Get weight value
                    let weight_idx = weight_offset + ic * weight_strides[0] + oc * weight_strides[1];

                    // Iterate over kernel
                    for k in 0..params.kernel_size {
                        // Calculate output position
                        let ol = il * params.stride + k * params.dilation;

                        // Adjust for padding
                        if ol < params.padding {
                            continue;
                        }
                        let ol_actual = ol - params.padding;

                        if ol_actual >= output_length {
                            continue;
                        }

                        // Get weight value for this kernel position
                        let weight_idx_k = weight_idx + k * weight_strides[2];
                        let weight_val = weight_storage[weight_idx_k];

                        // Accumulate to output
                        let output_idx = b * params.channels_output * output_length + oc * output_length + ol_actual;
                        result[output_idx] = result[output_idx] + input_val * weight_val;
                    }
                }
            }
        }
    }

    Ok(result)
}

// ConvTranspose2D
pub fn conv_transpose2d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    weight_storage: &[T],
    weight_layout: &Layout,
    params: &ParamsConvTranspose2D,
) -> HoduResult<Vec<T>> {
    // Input: [batch, in_channels, height, width]
    // Weight: [in_channels, out_channels, kernel_h, kernel_w]
    // Output: [batch, out_channels, out_height, out_width]

    let output_height = (params.input_height - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_height - 1)
        + params.output_padding
        + 1;
    let output_width = (params.input_width - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_width - 1)
        + params.output_padding
        + 1;

    let output_size = params.batch_size * params.channels_output * output_height * output_width;
    let mut result = vec![T::default(); output_size];

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    // Iterate over batch
    for b in 0..params.batch_size {
        // Iterate over input channels
        for ic in 0..params.channels_input {
            // Iterate over input height
            for ih in 0..params.input_height {
                // Iterate over input width
                for iw in 0..params.input_width {
                    // Get input value
                    let input_idx = input_offset
                        + b * input_strides[0]
                        + ic * input_strides[1]
                        + ih * input_strides[2]
                        + iw * input_strides[3];
                    let input_val = input_storage[input_idx];

                    // Iterate over output channels
                    for oc in 0..params.channels_output {
                        // Get base weight index
                        let weight_base_idx = weight_offset + ic * weight_strides[0] + oc * weight_strides[1];

                        // Iterate over kernel height
                        for kh in 0..params.kernel_height {
                            // Iterate over kernel width
                            for kw in 0..params.kernel_width {
                                // Calculate output position
                                let oh = ih * params.stride + kh * params.dilation;
                                let ow = iw * params.stride + kw * params.dilation;

                                // Adjust for padding
                                if oh < params.padding || ow < params.padding {
                                    continue;
                                }
                                let oh_actual = oh - params.padding;
                                let ow_actual = ow - params.padding;

                                if oh_actual >= output_height || ow_actual >= output_width {
                                    continue;
                                }

                                // Get weight value
                                let weight_idx = weight_base_idx + kh * weight_strides[2] + kw * weight_strides[3];
                                let weight_val = weight_storage[weight_idx];

                                // Accumulate to output
                                let output_idx = b * params.channels_output * output_height * output_width
                                    + oc * output_height * output_width
                                    + oh_actual * output_width
                                    + ow_actual;
                                result[output_idx] = result[output_idx] + input_val * weight_val;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

// ConvTranspose3D
pub fn conv_transpose3d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    weight_storage: &[T],
    weight_layout: &Layout,
    params: &ParamsConvTranspose3D,
) -> HoduResult<Vec<T>> {
    // Input: [batch, in_channels, depth, height, width]
    // Weight: [in_channels, out_channels, kernel_d, kernel_h, kernel_w]
    // Output: [batch, out_channels, out_depth, out_height, out_width]

    let output_depth = (params.input_depth - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_depth - 1)
        + params.output_padding
        + 1;
    let output_height = (params.input_height - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_height - 1)
        + params.output_padding
        + 1;
    let output_width = (params.input_width - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_width - 1)
        + params.output_padding
        + 1;

    let output_size = params.batch_size * params.channels_output * output_depth * output_height * output_width;
    let mut result = vec![T::default(); output_size];

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    // Iterate over batch
    for b in 0..params.batch_size {
        // Iterate over input channels
        for ic in 0..params.channels_input {
            // Iterate over input depth
            for id in 0..params.input_depth {
                // Iterate over input height
                for ih in 0..params.input_height {
                    // Iterate over input width
                    for iw in 0..params.input_width {
                        // Get input value
                        let input_idx = input_offset
                            + b * input_strides[0]
                            + ic * input_strides[1]
                            + id * input_strides[2]
                            + ih * input_strides[3]
                            + iw * input_strides[4];
                        let input_val = input_storage[input_idx];

                        // Iterate over output channels
                        for oc in 0..params.channels_output {
                            // Get base weight index
                            let weight_base_idx = weight_offset + ic * weight_strides[0] + oc * weight_strides[1];

                            // Iterate over kernel depth
                            for kd in 0..params.kernel_depth {
                                // Iterate over kernel height
                                for kh in 0..params.kernel_height {
                                    // Iterate over kernel width
                                    for kw in 0..params.kernel_width {
                                        // Calculate output position
                                        let od = id * params.stride + kd * params.dilation;
                                        let oh = ih * params.stride + kh * params.dilation;
                                        let ow = iw * params.stride + kw * params.dilation;

                                        // Adjust for padding
                                        if od < params.padding || oh < params.padding || ow < params.padding {
                                            continue;
                                        }
                                        let od_actual = od - params.padding;
                                        let oh_actual = oh - params.padding;
                                        let ow_actual = ow - params.padding;

                                        if od_actual >= output_depth
                                            || oh_actual >= output_height
                                            || ow_actual >= output_width
                                        {
                                            continue;
                                        }

                                        // Get weight value
                                        let weight_idx = weight_base_idx
                                            + kd * weight_strides[2]
                                            + kh * weight_strides[3]
                                            + kw * weight_strides[4];
                                        let weight_val = weight_storage[weight_idx];

                                        // Accumulate to output
                                        let output_idx =
                                            b * params.channels_output * output_depth * output_height * output_width
                                                + oc * output_depth * output_height * output_width
                                                + od_actual * output_height * output_width
                                                + oh_actual * output_width
                                                + ow_actual;
                                        result[output_idx] = result[output_idx] + input_val * weight_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

// Convolution Gradient Weight operations

/// Conv1d weight gradient: computes gradient of weight given input and grad_output
/// Input: [batch, in_channels, length]
/// GradOutput: [batch, out_channels, output_length]
/// Weight gradient: [out_channels, in_channels, kernel_size]
pub fn conv1d_grad_weight_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    grad_output_storage: &[T],
    grad_output_layout: &Layout,
    params: &ParamsConv1D,
) -> HoduResult<Vec<T>> {
    let output_length =
        (params.length_input + 2 * params.padding - params.dilation * (params.kernel_size - 1) - 1) / params.stride + 1;

    // Weight gradient shape: [out_channels, in_channels, kernel_size]
    let weight_grad_size = params.channels_output * params.channels_input * params.kernel_size;
    let mut weight_grad = vec![T::default(); weight_grad_size];

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    // Iterate over output channels
    for oc in 0..params.channels_output {
        // Iterate over input channels
        for ic in 0..params.channels_input {
            // Iterate over kernel positions
            for k in 0..params.kernel_size {
                let mut sum = T::default();

                // Iterate over batch
                for b in 0..params.batch_size {
                    // Iterate over output positions
                    for ol in 0..output_length {
                        // Calculate input position
                        let il = ol * params.stride + k * params.dilation;

                        // Check if position is within padded input
                        if il < params.padding || il >= params.length_input + params.padding {
                            continue; // Zero padding
                        }

                        let il_actual = il - params.padding;

                        // Get input value
                        let input_idx =
                            input_offset + b * input_strides[0] + ic * input_strides[1] + il_actual * input_strides[2];
                        let input_val = input_storage[input_idx];

                        // Get grad_output value
                        let grad_output_idx = grad_output_offset
                            + b * grad_output_strides[0]
                            + oc * grad_output_strides[1]
                            + ol * grad_output_strides[2];
                        let grad_output_val = grad_output_storage[grad_output_idx];

                        sum = sum + input_val * grad_output_val;
                    }
                }

                // Store weight gradient
                let weight_grad_idx = oc * params.channels_input * params.kernel_size + ic * params.kernel_size + k;
                weight_grad[weight_grad_idx] = sum;
            }
        }
    }

    Ok(weight_grad)
}

/// Conv2d weight gradient
/// Input: [batch, in_channels, height, width]
/// GradOutput: [batch, out_channels, out_height, out_width]
/// Weight gradient: [out_channels, in_channels, kernel_h, kernel_w]
pub fn conv2d_grad_weight_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    grad_output_storage: &[T],
    grad_output_layout: &Layout,
    params: &ParamsConv2D,
) -> HoduResult<Vec<T>> {
    let output_height = (params.input_height + 2 * params.padding - params.dilation * (params.kernel_height - 1) - 1)
        / params.stride
        + 1;
    let output_width =
        (params.input_width + 2 * params.padding - params.dilation * (params.kernel_width - 1) - 1) / params.stride + 1;

    // Weight gradient shape: [out_channels, in_channels, kernel_h, kernel_w]
    let weight_grad_size = params.channels_output * params.channels_input * params.kernel_height * params.kernel_width;
    let mut weight_grad = vec![T::default(); weight_grad_size];

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    // Iterate over output channels
    for oc in 0..params.channels_output {
        // Iterate over input channels
        for ic in 0..params.channels_input {
            // Iterate over kernel height
            for kh in 0..params.kernel_height {
                // Iterate over kernel width
                for kw in 0..params.kernel_width {
                    let mut sum = T::default();

                    // Iterate over batch
                    for b in 0..params.batch_size {
                        // Iterate over output height
                        for oh in 0..output_height {
                            // Iterate over output width
                            for ow in 0..output_width {
                                // Calculate input position
                                let ih = oh * params.stride + kh * params.dilation;
                                let iw = ow * params.stride + kw * params.dilation;

                                // Check bounds with padding
                                if ih < params.padding
                                    || ih >= params.input_height + params.padding
                                    || iw < params.padding
                                    || iw >= params.input_width + params.padding
                                {
                                    continue; // Zero padding
                                }

                                let ih_actual = ih - params.padding;
                                let iw_actual = iw - params.padding;

                                // Get input value
                                let input_idx = input_offset
                                    + b * input_strides[0]
                                    + ic * input_strides[1]
                                    + ih_actual * input_strides[2]
                                    + iw_actual * input_strides[3];
                                let input_val = input_storage[input_idx];

                                // Get grad_output value
                                let grad_output_idx = grad_output_offset
                                    + b * grad_output_strides[0]
                                    + oc * grad_output_strides[1]
                                    + oh * grad_output_strides[2]
                                    + ow * grad_output_strides[3];
                                let grad_output_val = grad_output_storage[grad_output_idx];

                                sum = sum + input_val * grad_output_val;
                            }
                        }
                    }

                    // Store weight gradient
                    let weight_grad_idx = oc * params.channels_input * params.kernel_height * params.kernel_width
                        + ic * params.kernel_height * params.kernel_width
                        + kh * params.kernel_width
                        + kw;
                    weight_grad[weight_grad_idx] = sum;
                }
            }
        }
    }

    Ok(weight_grad)
}

/// Conv3d weight gradient
/// Input: [batch, in_channels, depth, height, width]
/// GradOutput: [batch, out_channels, out_depth, out_height, out_width]
/// Weight gradient: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
pub fn conv3d_grad_weight_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    grad_output_storage: &[T],
    grad_output_layout: &Layout,
    params: &ParamsConv3D,
) -> HoduResult<Vec<T>> {
    let output_depth =
        (params.input_depth + 2 * params.padding - params.dilation * (params.kernel_depth - 1) - 1) / params.stride + 1;
    let output_height = (params.input_height + 2 * params.padding - params.dilation * (params.kernel_height - 1) - 1)
        / params.stride
        + 1;
    let output_width =
        (params.input_width + 2 * params.padding - params.dilation * (params.kernel_width - 1) - 1) / params.stride + 1;

    // Weight gradient shape: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
    let weight_grad_size = params.channels_output
        * params.channels_input
        * params.kernel_depth
        * params.kernel_height
        * params.kernel_width;
    let mut weight_grad = vec![T::default(); weight_grad_size];

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    // Iterate over output channels
    for oc in 0..params.channels_output {
        // Iterate over input channels
        for ic in 0..params.channels_input {
            // Iterate over kernel depth
            for kd in 0..params.kernel_depth {
                // Iterate over kernel height
                for kh in 0..params.kernel_height {
                    // Iterate over kernel width
                    for kw in 0..params.kernel_width {
                        let mut sum = T::default();

                        // Iterate over batch
                        for b in 0..params.batch_size {
                            // Iterate over output depth
                            for od in 0..output_depth {
                                // Iterate over output height
                                for oh in 0..output_height {
                                    // Iterate over output width
                                    for ow in 0..output_width {
                                        // Calculate input position
                                        let id = od * params.stride + kd * params.dilation;
                                        let ih = oh * params.stride + kh * params.dilation;
                                        let iw = ow * params.stride + kw * params.dilation;

                                        // Check bounds with padding
                                        if id < params.padding
                                            || id >= params.input_depth + params.padding
                                            || ih < params.padding
                                            || ih >= params.input_height + params.padding
                                            || iw < params.padding
                                            || iw >= params.input_width + params.padding
                                        {
                                            continue; // Zero padding
                                        }

                                        let id_actual = id - params.padding;
                                        let ih_actual = ih - params.padding;
                                        let iw_actual = iw - params.padding;

                                        // Get input value
                                        let input_idx = input_offset
                                            + b * input_strides[0]
                                            + ic * input_strides[1]
                                            + id_actual * input_strides[2]
                                            + ih_actual * input_strides[3]
                                            + iw_actual * input_strides[4];
                                        let input_val = input_storage[input_idx];

                                        // Get grad_output value
                                        let grad_output_idx = grad_output_offset
                                            + b * grad_output_strides[0]
                                            + oc * grad_output_strides[1]
                                            + od * grad_output_strides[2]
                                            + oh * grad_output_strides[3]
                                            + ow * grad_output_strides[4];
                                        let grad_output_val = grad_output_storage[grad_output_idx];

                                        sum = sum + input_val * grad_output_val;
                                    }
                                }
                            }
                        }

                        // Store weight gradient
                        let weight_grad_idx = oc
                            * params.channels_input
                            * params.kernel_depth
                            * params.kernel_height
                            * params.kernel_width
                            + ic * params.kernel_depth * params.kernel_height * params.kernel_width
                            + kd * params.kernel_height * params.kernel_width
                            + kh * params.kernel_width
                            + kw;
                        weight_grad[weight_grad_idx] = sum;
                    }
                }
            }
        }
    }

    Ok(weight_grad)
}

// ConvTranspose Gradient Weight operations

/// ConvTranspose1d weight gradient
/// Input: [batch, in_channels, length_in]
/// GradOutput: [batch, out_channels, length_out]
/// Weight gradient: [in_channels, out_channels, kernel_size]
pub fn conv_transpose1d_grad_weight_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    grad_output_storage: &[T],
    grad_output_layout: &Layout,
    params: &ParamsConvTranspose1D,
) -> HoduResult<Vec<T>> {
    let output_length = (params.length_input - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_size - 1)
        + params.output_padding
        + 1;

    // Weight gradient shape: [in_channels, out_channels, kernel_size]
    let weight_grad_size = params.channels_input * params.channels_output * params.kernel_size;
    let mut weight_grad = vec![T::default(); weight_grad_size];

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    // Iterate over input channels (first dimension of weight in ConvTranspose)
    for ic in 0..params.channels_input {
        // Iterate over output channels
        for oc in 0..params.channels_output {
            // Iterate over kernel positions
            for k in 0..params.kernel_size {
                let mut sum = T::default();

                // Iterate over batch
                for b in 0..params.batch_size {
                    // Iterate over input positions
                    for il in 0..params.length_input {
                        // For each input position, find corresponding output positions
                        let out_start = il * params.stride + k * params.dilation;

                        // Check if this output position exists (with padding)
                        if out_start < params.padding || out_start >= output_length + params.padding {
                            continue;
                        }

                        let ol = out_start - params.padding;

                        // Get input value
                        let input_idx =
                            input_offset + b * input_strides[0] + ic * input_strides[1] + il * input_strides[2];
                        let input_val = input_storage[input_idx];

                        // Get grad_output value
                        let grad_output_idx = grad_output_offset
                            + b * grad_output_strides[0]
                            + oc * grad_output_strides[1]
                            + ol * grad_output_strides[2];
                        let grad_output_val = grad_output_storage[grad_output_idx];

                        sum = sum + input_val * grad_output_val;
                    }
                }

                // Store weight gradient
                let weight_grad_idx = ic * params.channels_output * params.kernel_size + oc * params.kernel_size + k;
                weight_grad[weight_grad_idx] = sum;
            }
        }
    }

    Ok(weight_grad)
}

/// ConvTranspose2d weight gradient
/// Input: [batch, in_channels, height_in, width_in]
/// GradOutput: [batch, out_channels, height_out, width_out]
/// Weight gradient: [in_channels, out_channels, kernel_h, kernel_w]
pub fn conv_transpose2d_grad_weight_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    grad_output_storage: &[T],
    grad_output_layout: &Layout,
    params: &ParamsConvTranspose2D,
) -> HoduResult<Vec<T>> {
    let output_height = (params.input_height - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_height - 1)
        + params.output_padding
        + 1;
    let output_width = (params.input_width - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_width - 1)
        + params.output_padding
        + 1;

    // Weight gradient shape: [in_channels, out_channels, kernel_h, kernel_w]
    let weight_grad_size = params.channels_input * params.channels_output * params.kernel_height * params.kernel_width;
    let mut weight_grad = vec![T::default(); weight_grad_size];

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    // Iterate over input channels
    for ic in 0..params.channels_input {
        // Iterate over output channels
        for oc in 0..params.channels_output {
            // Iterate over kernel height
            for kh in 0..params.kernel_height {
                // Iterate over kernel width
                for kw in 0..params.kernel_width {
                    let mut sum = T::default();

                    // Iterate over batch
                    for b in 0..params.batch_size {
                        // Iterate over input height
                        for ih in 0..params.input_height {
                            // Iterate over input width
                            for iw in 0..params.input_width {
                                // Calculate output position
                                let oh_before_pad = ih * params.stride + kh * params.dilation;
                                let ow_before_pad = iw * params.stride + kw * params.dilation;

                                // Check bounds with padding
                                if oh_before_pad < params.padding
                                    || oh_before_pad >= output_height + params.padding
                                    || ow_before_pad < params.padding
                                    || ow_before_pad >= output_width + params.padding
                                {
                                    continue;
                                }

                                let oh = oh_before_pad - params.padding;
                                let ow = ow_before_pad - params.padding;

                                // Get input value
                                let input_idx = input_offset
                                    + b * input_strides[0]
                                    + ic * input_strides[1]
                                    + ih * input_strides[2]
                                    + iw * input_strides[3];
                                let input_val = input_storage[input_idx];

                                // Get grad_output value
                                let grad_output_idx = grad_output_offset
                                    + b * grad_output_strides[0]
                                    + oc * grad_output_strides[1]
                                    + oh * grad_output_strides[2]
                                    + ow * grad_output_strides[3];
                                let grad_output_val = grad_output_storage[grad_output_idx];

                                sum = sum + input_val * grad_output_val;
                            }
                        }
                    }

                    // Store weight gradient
                    let weight_grad_idx = ic * params.channels_output * params.kernel_height * params.kernel_width
                        + oc * params.kernel_height * params.kernel_width
                        + kh * params.kernel_width
                        + kw;
                    weight_grad[weight_grad_idx] = sum;
                }
            }
        }
    }

    Ok(weight_grad)
}

/// ConvTranspose3d weight gradient
/// Input: [batch, in_channels, depth_in, height_in, width_in]
/// GradOutput: [batch, out_channels, depth_out, height_out, width_out]
/// Weight gradient: [in_channels, out_channels, kernel_d, kernel_h, kernel_w]
pub fn conv_transpose3d_grad_weight_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T>>(
    input_storage: &[T],
    input_layout: &Layout,
    grad_output_storage: &[T],
    grad_output_layout: &Layout,
    params: &ParamsConvTranspose3D,
) -> HoduResult<Vec<T>> {
    let output_depth = (params.input_depth - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_depth - 1)
        + params.output_padding
        + 1;
    let output_height = (params.input_height - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_height - 1)
        + params.output_padding
        + 1;
    let output_width = (params.input_width - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_width - 1)
        + params.output_padding
        + 1;

    // Weight gradient shape: [in_channels, out_channels, kernel_d, kernel_h, kernel_w]
    let weight_grad_size = params.channels_input
        * params.channels_output
        * params.kernel_depth
        * params.kernel_height
        * params.kernel_width;
    let mut weight_grad = vec![T::default(); weight_grad_size];

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    // Iterate over input channels
    for ic in 0..params.channels_input {
        // Iterate over output channels
        for oc in 0..params.channels_output {
            // Iterate over kernel depth
            for kd in 0..params.kernel_depth {
                // Iterate over kernel height
                for kh in 0..params.kernel_height {
                    // Iterate over kernel width
                    for kw in 0..params.kernel_width {
                        let mut sum = T::default();

                        // Iterate over batch
                        for b in 0..params.batch_size {
                            // Iterate over input depth
                            for id in 0..params.input_depth {
                                // Iterate over input height
                                for ih in 0..params.input_height {
                                    // Iterate over input width
                                    for iw in 0..params.input_width {
                                        // Calculate output position
                                        let od_before_pad = id * params.stride + kd * params.dilation;
                                        let oh_before_pad = ih * params.stride + kh * params.dilation;
                                        let ow_before_pad = iw * params.stride + kw * params.dilation;

                                        // Check bounds with padding
                                        if od_before_pad < params.padding
                                            || od_before_pad >= output_depth + params.padding
                                            || oh_before_pad < params.padding
                                            || oh_before_pad >= output_height + params.padding
                                            || ow_before_pad < params.padding
                                            || ow_before_pad >= output_width + params.padding
                                        {
                                            continue;
                                        }

                                        let od = od_before_pad - params.padding;
                                        let oh = oh_before_pad - params.padding;
                                        let ow = ow_before_pad - params.padding;

                                        // Get input value
                                        let input_idx = input_offset
                                            + b * input_strides[0]
                                            + ic * input_strides[1]
                                            + id * input_strides[2]
                                            + ih * input_strides[3]
                                            + iw * input_strides[4];
                                        let input_val = input_storage[input_idx];

                                        // Get grad_output value
                                        let grad_output_idx = grad_output_offset
                                            + b * grad_output_strides[0]
                                            + oc * grad_output_strides[1]
                                            + od * grad_output_strides[2]
                                            + oh * grad_output_strides[3]
                                            + ow * grad_output_strides[4];
                                        let grad_output_val = grad_output_storage[grad_output_idx];

                                        sum = sum + input_val * grad_output_val;
                                    }
                                }
                            }
                        }

                        // Store weight gradient
                        let weight_grad_idx = ic
                            * params.channels_output
                            * params.kernel_depth
                            * params.kernel_height
                            * params.kernel_width
                            + oc * params.kernel_depth * params.kernel_height * params.kernel_width
                            + kd * params.kernel_height * params.kernel_width
                            + kh * params.kernel_width
                            + kw;
                        weight_grad[weight_grad_idx] = sum;
                    }
                }
            }
        }
    }

    Ok(weight_grad)
}
