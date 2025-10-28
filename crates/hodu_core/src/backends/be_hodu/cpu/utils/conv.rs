use crate::{
    backends::op::conv::{
        ParamsConv1D, ParamsConv2D, ParamsConv3D, ParamsConvTranspose1D, ParamsConvTranspose2D, ParamsConvTranspose3D,
    },
    compat::*,
    error::HoduResult,
    types::layout::Layout,
};

// Conv1D
pub fn conv1d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T> + Send + Sync>(
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

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    #[cfg(feature = "rayon")]
    {
        let results: Vec<T> = (0..params.batch_size)
            .into_par_iter()
            .flat_map(|b| {
                (0..params.channels_output)
                    .into_par_iter()
                    .flat_map(|oc| {
                        (0..output_length)
                            .into_par_iter()
                            .map(|ol| {
                                let mut sum = T::default();

                                for ic in 0..params.channels_input {
                                    for k in 0..params.kernel_size {
                                        let il = ol * params.stride + k * params.dilation;

                                        if il < params.padding || il >= params.length_input + params.padding {
                                            continue;
                                        }

                                        let il_actual = il - params.padding;

                                        let input_idx = input_offset
                                            + b * input_strides[0]
                                            + ic * input_strides[1]
                                            + il_actual * input_strides[2];
                                        let input_val = input_storage[input_idx];

                                        let weight_idx = weight_offset
                                            + oc * weight_strides[0]
                                            + ic * weight_strides[1]
                                            + k * weight_strides[2];
                                        let weight_val = weight_storage[weight_idx];

                                        sum = sum + input_val * weight_val;
                                    }
                                }

                                sum
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(results)
    }

    #[cfg(not(feature = "rayon"))]
    {
        let output_size = params.batch_size * params.channels_output * output_length;
        let mut result = vec![T::default(); output_size];

        for b in 0..params.batch_size {
            for oc in 0..params.channels_output {
                for ol in 0..output_length {
                    let mut sum = T::default();

                    for ic in 0..params.channels_input {
                        for k in 0..params.kernel_size {
                            let il = ol * params.stride + k * params.dilation;

                            if il < params.padding || il >= params.length_input + params.padding {
                                continue;
                            }

                            let il_actual = il - params.padding;

                            let input_idx = input_offset
                                + b * input_strides[0]
                                + ic * input_strides[1]
                                + il_actual * input_strides[2];
                            let input_val = input_storage[input_idx];

                            let weight_idx =
                                weight_offset + oc * weight_strides[0] + ic * weight_strides[1] + k * weight_strides[2];
                            let weight_val = weight_storage[weight_idx];

                            sum = sum + input_val * weight_val;
                        }
                    }

                    let output_idx = b * params.channels_output * output_length + oc * output_length + ol;
                    result[output_idx] = sum;
                }
            }
        }

        Ok(result)
    }
}

// Conv2D
pub fn conv2d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T> + Send + Sync>(
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

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    #[cfg(feature = "rayon")]
    {
        let results: Vec<T> = (0..params.batch_size)
            .into_par_iter()
            .flat_map(|b| {
                (0..params.channels_output)
                    .into_par_iter()
                    .flat_map(|oc| {
                        (0..output_height)
                            .into_par_iter()
                            .flat_map(|oh| {
                                (0..output_width)
                                    .into_par_iter()
                                    .map(|ow| {
                                        let mut sum = T::default();

                                        for ic in 0..params.channels_input {
                                            for kh in 0..params.kernel_height {
                                                for kw in 0..params.kernel_width {
                                                    let ih = oh * params.stride + kh * params.dilation;
                                                    let iw = ow * params.stride + kw * params.dilation;

                                                    if ih < params.padding
                                                        || ih >= params.input_height + params.padding
                                                        || iw < params.padding
                                                        || iw >= params.input_width + params.padding
                                                    {
                                                        continue;
                                                    }

                                                    let ih_actual = ih - params.padding;
                                                    let iw_actual = iw - params.padding;

                                                    let input_idx = input_offset
                                                        + b * input_strides[0]
                                                        + ic * input_strides[1]
                                                        + ih_actual * input_strides[2]
                                                        + iw_actual * input_strides[3];
                                                    let input_val = input_storage[input_idx];

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

                                        sum
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(results)
    }

    #[cfg(not(feature = "rayon"))]
    {
        let output_size = params.batch_size * params.channels_output * output_height * output_width;
        let mut result = vec![T::default(); output_size];

        for b in 0..params.batch_size {
            for oc in 0..params.channels_output {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let mut sum = T::default();

                        for ic in 0..params.channels_input {
                            for kh in 0..params.kernel_height {
                                for kw in 0..params.kernel_width {
                                    let ih = oh * params.stride + kh * params.dilation;
                                    let iw = ow * params.stride + kw * params.dilation;

                                    if ih < params.padding
                                        || ih >= params.input_height + params.padding
                                        || iw < params.padding
                                        || iw >= params.input_width + params.padding
                                    {
                                        continue;
                                    }

                                    let ih_actual = ih - params.padding;
                                    let iw_actual = iw - params.padding;

                                    let input_idx = input_offset
                                        + b * input_strides[0]
                                        + ic * input_strides[1]
                                        + ih_actual * input_strides[2]
                                        + iw_actual * input_strides[3];
                                    let input_val = input_storage[input_idx];

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
}

// Conv3D
pub fn conv3d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T> + Send + Sync>(
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

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    #[cfg(feature = "rayon")]
    {
        let results: Vec<T> = (0..params.batch_size)
            .into_par_iter()
            .flat_map(|b| {
                (0..params.channels_output)
                    .into_par_iter()
                    .flat_map(|oc| {
                        (0..output_depth)
                            .into_par_iter()
                            .flat_map(|od| {
                                (0..output_height)
                                    .into_par_iter()
                                    .flat_map(|oh| {
                                        (0..output_width)
                                            .into_par_iter()
                                            .map(|ow| {
                                                let mut sum = T::default();

                                                for ic in 0..params.channels_input {
                                                    for kd in 0..params.kernel_depth {
                                                        for kh in 0..params.kernel_height {
                                                            for kw in 0..params.kernel_width {
                                                                let id = od * params.stride + kd * params.dilation;
                                                                let ih = oh * params.stride + kh * params.dilation;
                                                                let iw = ow * params.stride + kw * params.dilation;

                                                                if id < params.padding
                                                                    || id >= params.input_depth + params.padding
                                                                    || ih < params.padding
                                                                    || ih >= params.input_height + params.padding
                                                                    || iw < params.padding
                                                                    || iw >= params.input_width + params.padding
                                                                {
                                                                    continue;
                                                                }

                                                                let id_actual = id - params.padding;
                                                                let ih_actual = ih - params.padding;
                                                                let iw_actual = iw - params.padding;

                                                                let input_idx = input_offset
                                                                    + b * input_strides[0]
                                                                    + ic * input_strides[1]
                                                                    + id_actual * input_strides[2]
                                                                    + ih_actual * input_strides[3]
                                                                    + iw_actual * input_strides[4];
                                                                let input_val = input_storage[input_idx];

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

                                                sum
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(results)
    }

    #[cfg(not(feature = "rayon"))]
    {
        let output_size = params.batch_size * params.channels_output * output_depth * output_height * output_width;
        let mut result = vec![T::default(); output_size];

        for b in 0..params.batch_size {
            for oc in 0..params.channels_output {
                for od in 0..output_depth {
                    for oh in 0..output_height {
                        for ow in 0..output_width {
                            let mut sum = T::default();

                            for ic in 0..params.channels_input {
                                for kd in 0..params.kernel_depth {
                                    for kh in 0..params.kernel_height {
                                        for kw in 0..params.kernel_width {
                                            let id = od * params.stride + kd * params.dilation;
                                            let ih = oh * params.stride + kh * params.dilation;
                                            let iw = ow * params.stride + kw * params.dilation;

                                            if id < params.padding
                                                || id >= params.input_depth + params.padding
                                                || ih < params.padding
                                                || ih >= params.input_height + params.padding
                                                || iw < params.padding
                                                || iw >= params.input_width + params.padding
                                            {
                                                continue;
                                            }

                                            let id_actual = id - params.padding;
                                            let ih_actual = ih - params.padding;
                                            let iw_actual = iw - params.padding;

                                            let input_idx = input_offset
                                                + b * input_strides[0]
                                                + ic * input_strides[1]
                                                + id_actual * input_strides[2]
                                                + ih_actual * input_strides[3]
                                                + iw_actual * input_strides[4];
                                            let input_val = input_storage[input_idx];

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
}

// ConvTranspose1D - Parallelize at batch level due to output accumulation
pub fn conv_transpose1d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T> + Send + Sync>(
    input_storage: &[T],
    input_layout: &Layout,
    weight_storage: &[T],
    weight_layout: &Layout,
    params: &ParamsConvTranspose1D,
) -> HoduResult<Vec<T>> {
    // Input: [batch, in_channels, length]
    // Weight: [in_channels, out_channels, kernel_size]
    // Output: [batch, out_channels, output_length]

    let output_length = (params.length_input - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_size - 1)
        + params.output_padding
        + 1;

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    #[cfg(feature = "rayon")]
    {
        let batch_results: Vec<Vec<T>> = (0..params.batch_size)
            .into_par_iter()
            .map(|b| {
                let mut batch_result = vec![T::default(); params.channels_output * output_length];

                for ic in 0..params.channels_input {
                    for il in 0..params.length_input {
                        let input_idx =
                            input_offset + b * input_strides[0] + ic * input_strides[1] + il * input_strides[2];
                        let input_val = input_storage[input_idx];

                        for oc in 0..params.channels_output {
                            let weight_idx = weight_offset + ic * weight_strides[0] + oc * weight_strides[1];

                            for k in 0..params.kernel_size {
                                let ol = il * params.stride + k * params.dilation;

                                if ol < params.padding {
                                    continue;
                                }
                                let ol_actual = ol - params.padding;

                                if ol_actual >= output_length {
                                    continue;
                                }

                                let weight_idx_k = weight_idx + k * weight_strides[2];
                                let weight_val = weight_storage[weight_idx_k];

                                let local_idx = oc * output_length + ol_actual;
                                batch_result[local_idx] = batch_result[local_idx] + input_val * weight_val;
                            }
                        }
                    }
                }

                batch_result
            })
            .collect();

        let result: Vec<T> = batch_results.into_iter().flatten().collect();
        Ok(result)
    }

    #[cfg(not(feature = "rayon"))]
    {
        let output_size = params.batch_size * params.channels_output * output_length;
        let mut result = vec![T::default(); output_size];

        for b in 0..params.batch_size {
            for ic in 0..params.channels_input {
                for il in 0..params.length_input {
                    let input_idx = input_offset + b * input_strides[0] + ic * input_strides[1] + il * input_strides[2];
                    let input_val = input_storage[input_idx];

                    for oc in 0..params.channels_output {
                        let weight_idx = weight_offset + ic * weight_strides[0] + oc * weight_strides[1];

                        for k in 0..params.kernel_size {
                            let ol = il * params.stride + k * params.dilation;

                            if ol < params.padding {
                                continue;
                            }
                            let ol_actual = ol - params.padding;

                            if ol_actual >= output_length {
                                continue;
                            }

                            let weight_idx_k = weight_idx + k * weight_strides[2];
                            let weight_val = weight_storage[weight_idx_k];

                            let output_idx =
                                b * params.channels_output * output_length + oc * output_length + ol_actual;
                            result[output_idx] = result[output_idx] + input_val * weight_val;
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

// ConvTranspose2D - Parallelize at batch level
pub fn conv_transpose2d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T> + Send + Sync>(
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

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    #[cfg(feature = "rayon")]
    {
        let batch_results: Vec<Vec<T>> = (0..params.batch_size)
            .into_par_iter()
            .map(|b| {
                let mut batch_result = vec![T::default(); params.channels_output * output_height * output_width];

                for ic in 0..params.channels_input {
                    for ih in 0..params.input_height {
                        for iw in 0..params.input_width {
                            let input_idx = input_offset
                                + b * input_strides[0]
                                + ic * input_strides[1]
                                + ih * input_strides[2]
                                + iw * input_strides[3];
                            let input_val = input_storage[input_idx];

                            for oc in 0..params.channels_output {
                                let weight_base_idx = weight_offset + ic * weight_strides[0] + oc * weight_strides[1];

                                for kh in 0..params.kernel_height {
                                    for kw in 0..params.kernel_width {
                                        let oh = ih * params.stride + kh * params.dilation;
                                        let ow = iw * params.stride + kw * params.dilation;

                                        if oh < params.padding || ow < params.padding {
                                            continue;
                                        }
                                        let oh_actual = oh - params.padding;
                                        let ow_actual = ow - params.padding;

                                        if oh_actual >= output_height || ow_actual >= output_width {
                                            continue;
                                        }

                                        let weight_idx =
                                            weight_base_idx + kh * weight_strides[2] + kw * weight_strides[3];
                                        let weight_val = weight_storage[weight_idx];

                                        let local_idx =
                                            oc * output_height * output_width + oh_actual * output_width + ow_actual;
                                        batch_result[local_idx] = batch_result[local_idx] + input_val * weight_val;
                                    }
                                }
                            }
                        }
                    }
                }

                batch_result
            })
            .collect();

        let result: Vec<T> = batch_results.into_iter().flatten().collect();
        Ok(result)
    }

    #[cfg(not(feature = "rayon"))]
    {
        let output_size = params.batch_size * params.channels_output * output_height * output_width;
        let mut result = vec![T::default(); output_size];

        for b in 0..params.batch_size {
            for ic in 0..params.channels_input {
                for ih in 0..params.input_height {
                    for iw in 0..params.input_width {
                        let input_idx = input_offset
                            + b * input_strides[0]
                            + ic * input_strides[1]
                            + ih * input_strides[2]
                            + iw * input_strides[3];
                        let input_val = input_storage[input_idx];

                        for oc in 0..params.channels_output {
                            let weight_base_idx = weight_offset + ic * weight_strides[0] + oc * weight_strides[1];

                            for kh in 0..params.kernel_height {
                                for kw in 0..params.kernel_width {
                                    let oh = ih * params.stride + kh * params.dilation;
                                    let ow = iw * params.stride + kw * params.dilation;

                                    if oh < params.padding || ow < params.padding {
                                        continue;
                                    }
                                    let oh_actual = oh - params.padding;
                                    let ow_actual = ow - params.padding;

                                    if oh_actual >= output_height || ow_actual >= output_width {
                                        continue;
                                    }

                                    let weight_idx = weight_base_idx + kh * weight_strides[2] + kw * weight_strides[3];
                                    let weight_val = weight_storage[weight_idx];

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
}

// ConvTranspose3D - Parallelize at batch level
pub fn conv_transpose3d_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T> + Send + Sync>(
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

    let input_offset = input_layout.get_offset();
    let weight_offset = weight_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let weight_strides = weight_layout.get_strides();

    #[cfg(feature = "rayon")]
    {
        let batch_results: Vec<Vec<T>> = (0..params.batch_size)
            .into_par_iter()
            .map(|b| {
                let mut batch_result =
                    vec![T::default(); params.channels_output * output_depth * output_height * output_width];

                for ic in 0..params.channels_input {
                    for id in 0..params.input_depth {
                        for ih in 0..params.input_height {
                            for iw in 0..params.input_width {
                                let input_idx = input_offset
                                    + b * input_strides[0]
                                    + ic * input_strides[1]
                                    + id * input_strides[2]
                                    + ih * input_strides[3]
                                    + iw * input_strides[4];
                                let input_val = input_storage[input_idx];

                                for oc in 0..params.channels_output {
                                    let weight_base_idx =
                                        weight_offset + ic * weight_strides[0] + oc * weight_strides[1];

                                    for kd in 0..params.kernel_depth {
                                        for kh in 0..params.kernel_height {
                                            for kw in 0..params.kernel_width {
                                                let od = id * params.stride + kd * params.dilation;
                                                let oh = ih * params.stride + kh * params.dilation;
                                                let ow = iw * params.stride + kw * params.dilation;

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

                                                let weight_idx = weight_base_idx
                                                    + kd * weight_strides[2]
                                                    + kh * weight_strides[3]
                                                    + kw * weight_strides[4];
                                                let weight_val = weight_storage[weight_idx];

                                                let local_idx = oc * output_depth * output_height * output_width
                                                    + od_actual * output_height * output_width
                                                    + oh_actual * output_width
                                                    + ow_actual;
                                                batch_result[local_idx] =
                                                    batch_result[local_idx] + input_val * weight_val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                batch_result
            })
            .collect();

        let result: Vec<T> = batch_results.into_iter().flatten().collect();
        Ok(result)
    }

    #[cfg(not(feature = "rayon"))]
    {
        let output_size = params.batch_size * params.channels_output * output_depth * output_height * output_width;
        let mut result = vec![T::default(); output_size];

        for b in 0..params.batch_size {
            for ic in 0..params.channels_input {
                for id in 0..params.input_depth {
                    for ih in 0..params.input_height {
                        for iw in 0..params.input_width {
                            let input_idx = input_offset
                                + b * input_strides[0]
                                + ic * input_strides[1]
                                + id * input_strides[2]
                                + ih * input_strides[3]
                                + iw * input_strides[4];
                            let input_val = input_storage[input_idx];

                            for oc in 0..params.channels_output {
                                let weight_base_idx = weight_offset + ic * weight_strides[0] + oc * weight_strides[1];

                                for kd in 0..params.kernel_depth {
                                    for kh in 0..params.kernel_height {
                                        for kw in 0..params.kernel_width {
                                            let od = id * params.stride + kd * params.dilation;
                                            let oh = ih * params.stride + kh * params.dilation;
                                            let ow = iw * params.stride + kw * params.dilation;

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

                                            let weight_idx = weight_base_idx
                                                + kd * weight_strides[2]
                                                + kh * weight_strides[3]
                                                + kw * weight_strides[4];
                                            let weight_val = weight_storage[weight_idx];

                                            let output_idx = b
                                                * params.channels_output
                                                * output_depth
                                                * output_height
                                                * output_width
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
}

// Convolution Gradient Weight operations

/// Conv1d weight gradient
pub fn conv1d_grad_weight_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T> + Send + Sync>(
    input_storage: &[T],
    input_layout: &Layout,
    grad_output_storage: &[T],
    grad_output_layout: &Layout,
    params: &ParamsConv1D,
) -> HoduResult<Vec<T>> {
    let output_length =
        (params.length_input + 2 * params.padding - params.dilation * (params.kernel_size - 1) - 1) / params.stride + 1;

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    #[cfg(feature = "rayon")]
    {
        let weight_grad: Vec<T> = (0..params.channels_output)
            .into_par_iter()
            .flat_map(|oc| {
                (0..params.channels_input)
                    .into_par_iter()
                    .flat_map(|ic| {
                        (0..params.kernel_size)
                            .into_par_iter()
                            .map(|k| {
                                let mut sum = T::default();

                                for b in 0..params.batch_size {
                                    for ol in 0..output_length {
                                        let il = ol * params.stride + k * params.dilation;

                                        if il < params.padding || il >= params.length_input + params.padding {
                                            continue;
                                        }

                                        let il_actual = il - params.padding;

                                        let input_idx = input_offset
                                            + b * input_strides[0]
                                            + ic * input_strides[1]
                                            + il_actual * input_strides[2];
                                        let input_val = input_storage[input_idx];

                                        let grad_output_idx = grad_output_offset
                                            + b * grad_output_strides[0]
                                            + oc * grad_output_strides[1]
                                            + ol * grad_output_strides[2];
                                        let grad_output_val = grad_output_storage[grad_output_idx];

                                        sum = sum + input_val * grad_output_val;
                                    }
                                }

                                sum
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(weight_grad)
    }

    #[cfg(not(feature = "rayon"))]
    {
        let weight_grad_size = params.channels_output * params.channels_input * params.kernel_size;
        let mut weight_grad = vec![T::default(); weight_grad_size];

        for oc in 0..params.channels_output {
            for ic in 0..params.channels_input {
                for k in 0..params.kernel_size {
                    let mut sum = T::default();

                    for b in 0..params.batch_size {
                        for ol in 0..output_length {
                            let il = ol * params.stride + k * params.dilation;

                            if il < params.padding || il >= params.length_input + params.padding {
                                continue;
                            }

                            let il_actual = il - params.padding;

                            let input_idx = input_offset
                                + b * input_strides[0]
                                + ic * input_strides[1]
                                + il_actual * input_strides[2];
                            let input_val = input_storage[input_idx];

                            let grad_output_idx = grad_output_offset
                                + b * grad_output_strides[0]
                                + oc * grad_output_strides[1]
                                + ol * grad_output_strides[2];
                            let grad_output_val = grad_output_storage[grad_output_idx];

                            sum = sum + input_val * grad_output_val;
                        }
                    }

                    let weight_grad_idx = oc * params.channels_input * params.kernel_size + ic * params.kernel_size + k;
                    weight_grad[weight_grad_idx] = sum;
                }
            }
        }

        Ok(weight_grad)
    }
}

/// Conv2d weight gradient
pub fn conv2d_grad_weight_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T> + Send + Sync>(
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

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    #[cfg(feature = "rayon")]
    {
        let weight_grad: Vec<T> = (0..params.channels_output)
            .into_par_iter()
            .flat_map(|oc| {
                (0..params.channels_input)
                    .into_par_iter()
                    .flat_map(|ic| {
                        (0..params.kernel_height)
                            .into_par_iter()
                            .flat_map(|kh| {
                                (0..params.kernel_width)
                                    .into_par_iter()
                                    .map(|kw| {
                                        let mut sum = T::default();

                                        for b in 0..params.batch_size {
                                            for oh in 0..output_height {
                                                for ow in 0..output_width {
                                                    let ih = oh * params.stride + kh * params.dilation;
                                                    let iw = ow * params.stride + kw * params.dilation;

                                                    if ih < params.padding
                                                        || ih >= params.input_height + params.padding
                                                        || iw < params.padding
                                                        || iw >= params.input_width + params.padding
                                                    {
                                                        continue;
                                                    }

                                                    let ih_actual = ih - params.padding;
                                                    let iw_actual = iw - params.padding;

                                                    let input_idx = input_offset
                                                        + b * input_strides[0]
                                                        + ic * input_strides[1]
                                                        + ih_actual * input_strides[2]
                                                        + iw_actual * input_strides[3];
                                                    let input_val = input_storage[input_idx];

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

                                        sum
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(weight_grad)
    }

    #[cfg(not(feature = "rayon"))]
    {
        let weight_grad_size =
            params.channels_output * params.channels_input * params.kernel_height * params.kernel_width;
        let mut weight_grad = vec![T::default(); weight_grad_size];

        for oc in 0..params.channels_output {
            for ic in 0..params.channels_input {
                for kh in 0..params.kernel_height {
                    for kw in 0..params.kernel_width {
                        let mut sum = T::default();

                        for b in 0..params.batch_size {
                            for oh in 0..output_height {
                                for ow in 0..output_width {
                                    let ih = oh * params.stride + kh * params.dilation;
                                    let iw = ow * params.stride + kw * params.dilation;

                                    if ih < params.padding
                                        || ih >= params.input_height + params.padding
                                        || iw < params.padding
                                        || iw >= params.input_width + params.padding
                                    {
                                        continue;
                                    }

                                    let ih_actual = ih - params.padding;
                                    let iw_actual = iw - params.padding;

                                    let input_idx = input_offset
                                        + b * input_strides[0]
                                        + ic * input_strides[1]
                                        + ih_actual * input_strides[2]
                                        + iw_actual * input_strides[3];
                                    let input_val = input_storage[input_idx];

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
}

/// Conv3d weight gradient
pub fn conv3d_grad_weight_map<T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T> + Send + Sync>(
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

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    #[cfg(feature = "rayon")]
    {
        let weight_grad: Vec<T> = (0..params.channels_output)
            .into_par_iter()
            .flat_map(|oc| {
                (0..params.channels_input)
                    .into_par_iter()
                    .flat_map(|ic| {
                        (0..params.kernel_depth)
                            .into_par_iter()
                            .flat_map(|kd| {
                                (0..params.kernel_height)
                                    .into_par_iter()
                                    .flat_map(|kh| {
                                        (0..params.kernel_width)
                                            .into_par_iter()
                                            .map(|kw| {
                                                let mut sum = T::default();

                                                for b in 0..params.batch_size {
                                                    for od in 0..output_depth {
                                                        for oh in 0..output_height {
                                                            for ow in 0..output_width {
                                                                let id = od * params.stride + kd * params.dilation;
                                                                let ih = oh * params.stride + kh * params.dilation;
                                                                let iw = ow * params.stride + kw * params.dilation;

                                                                if id < params.padding
                                                                    || id >= params.input_depth + params.padding
                                                                    || ih < params.padding
                                                                    || ih >= params.input_height + params.padding
                                                                    || iw < params.padding
                                                                    || iw >= params.input_width + params.padding
                                                                {
                                                                    continue;
                                                                }

                                                                let id_actual = id - params.padding;
                                                                let ih_actual = ih - params.padding;
                                                                let iw_actual = iw - params.padding;

                                                                let input_idx = input_offset
                                                                    + b * input_strides[0]
                                                                    + ic * input_strides[1]
                                                                    + id_actual * input_strides[2]
                                                                    + ih_actual * input_strides[3]
                                                                    + iw_actual * input_strides[4];
                                                                let input_val = input_storage[input_idx];

                                                                let grad_output_idx = grad_output_offset
                                                                    + b * grad_output_strides[0]
                                                                    + oc * grad_output_strides[1]
                                                                    + od * grad_output_strides[2]
                                                                    + oh * grad_output_strides[3]
                                                                    + ow * grad_output_strides[4];
                                                                let grad_output_val =
                                                                    grad_output_storage[grad_output_idx];

                                                                sum = sum + input_val * grad_output_val;
                                                            }
                                                        }
                                                    }
                                                }

                                                sum
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(weight_grad)
    }

    #[cfg(not(feature = "rayon"))]
    {
        let weight_grad_size = params.channels_output
            * params.channels_input
            * params.kernel_depth
            * params.kernel_height
            * params.kernel_width;
        let mut weight_grad = vec![T::default(); weight_grad_size];

        for oc in 0..params.channels_output {
            for ic in 0..params.channels_input {
                for kd in 0..params.kernel_depth {
                    for kh in 0..params.kernel_height {
                        for kw in 0..params.kernel_width {
                            let mut sum = T::default();

                            for b in 0..params.batch_size {
                                for od in 0..output_depth {
                                    for oh in 0..output_height {
                                        for ow in 0..output_width {
                                            let id = od * params.stride + kd * params.dilation;
                                            let ih = oh * params.stride + kh * params.dilation;
                                            let iw = ow * params.stride + kw * params.dilation;

                                            if id < params.padding
                                                || id >= params.input_depth + params.padding
                                                || ih < params.padding
                                                || ih >= params.input_height + params.padding
                                                || iw < params.padding
                                                || iw >= params.input_width + params.padding
                                            {
                                                continue;
                                            }

                                            let id_actual = id - params.padding;
                                            let ih_actual = ih - params.padding;
                                            let iw_actual = iw - params.padding;

                                            let input_idx = input_offset
                                                + b * input_strides[0]
                                                + ic * input_strides[1]
                                                + id_actual * input_strides[2]
                                                + ih_actual * input_strides[3]
                                                + iw_actual * input_strides[4];
                                            let input_val = input_storage[input_idx];

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
}

// ConvTranspose Gradient Weight operations

/// ConvTranspose1d weight gradient
pub fn conv_transpose1d_grad_weight_map<
    T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T> + Send + Sync,
>(
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

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    #[cfg(feature = "rayon")]
    {
        let weight_grad: Vec<T> = (0..params.channels_input)
            .into_par_iter()
            .flat_map(|ic| {
                (0..params.channels_output)
                    .into_par_iter()
                    .flat_map(|oc| {
                        (0..params.kernel_size)
                            .into_par_iter()
                            .map(|k| {
                                let mut sum = T::default();

                                for b in 0..params.batch_size {
                                    for il in 0..params.length_input {
                                        let out_start = il * params.stride + k * params.dilation;

                                        if out_start < params.padding || out_start >= output_length + params.padding {
                                            continue;
                                        }

                                        let ol = out_start - params.padding;

                                        let input_idx = input_offset
                                            + b * input_strides[0]
                                            + ic * input_strides[1]
                                            + il * input_strides[2];
                                        let input_val = input_storage[input_idx];

                                        let grad_output_idx = grad_output_offset
                                            + b * grad_output_strides[0]
                                            + oc * grad_output_strides[1]
                                            + ol * grad_output_strides[2];
                                        let grad_output_val = grad_output_storage[grad_output_idx];

                                        sum = sum + input_val * grad_output_val;
                                    }
                                }

                                sum
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(weight_grad)
    }

    #[cfg(not(feature = "rayon"))]
    {
        let weight_grad_size = params.channels_input * params.channels_output * params.kernel_size;
        let mut weight_grad = vec![T::default(); weight_grad_size];

        for ic in 0..params.channels_input {
            for oc in 0..params.channels_output {
                for k in 0..params.kernel_size {
                    let mut sum = T::default();

                    for b in 0..params.batch_size {
                        for il in 0..params.length_input {
                            let out_start = il * params.stride + k * params.dilation;

                            if out_start < params.padding || out_start >= output_length + params.padding {
                                continue;
                            }

                            let ol = out_start - params.padding;

                            let input_idx =
                                input_offset + b * input_strides[0] + ic * input_strides[1] + il * input_strides[2];
                            let input_val = input_storage[input_idx];

                            let grad_output_idx = grad_output_offset
                                + b * grad_output_strides[0]
                                + oc * grad_output_strides[1]
                                + ol * grad_output_strides[2];
                            let grad_output_val = grad_output_storage[grad_output_idx];

                            sum = sum + input_val * grad_output_val;
                        }
                    }

                    let weight_grad_idx =
                        ic * params.channels_output * params.kernel_size + oc * params.kernel_size + k;
                    weight_grad[weight_grad_idx] = sum;
                }
            }
        }

        Ok(weight_grad)
    }
}

/// ConvTranspose2d weight gradient
pub fn conv_transpose2d_grad_weight_map<
    T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T> + Send + Sync,
>(
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

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    #[cfg(feature = "rayon")]
    {
        let weight_grad: Vec<T> = (0..params.channels_input)
            .into_par_iter()
            .flat_map(|ic| {
                (0..params.channels_output)
                    .into_par_iter()
                    .flat_map(|oc| {
                        (0..params.kernel_height)
                            .into_par_iter()
                            .flat_map(|kh| {
                                (0..params.kernel_width)
                                    .into_par_iter()
                                    .map(|kw| {
                                        let mut sum = T::default();

                                        for b in 0..params.batch_size {
                                            for ih in 0..params.input_height {
                                                for iw in 0..params.input_width {
                                                    let oh_before_pad = ih * params.stride + kh * params.dilation;
                                                    let ow_before_pad = iw * params.stride + kw * params.dilation;

                                                    if oh_before_pad < params.padding
                                                        || oh_before_pad >= output_height + params.padding
                                                        || ow_before_pad < params.padding
                                                        || ow_before_pad >= output_width + params.padding
                                                    {
                                                        continue;
                                                    }

                                                    let oh = oh_before_pad - params.padding;
                                                    let ow = ow_before_pad - params.padding;

                                                    let input_idx = input_offset
                                                        + b * input_strides[0]
                                                        + ic * input_strides[1]
                                                        + ih * input_strides[2]
                                                        + iw * input_strides[3];
                                                    let input_val = input_storage[input_idx];

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

                                        sum
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(weight_grad)
    }

    #[cfg(not(feature = "rayon"))]
    {
        let weight_grad_size =
            params.channels_input * params.channels_output * params.kernel_height * params.kernel_width;
        let mut weight_grad = vec![T::default(); weight_grad_size];

        for ic in 0..params.channels_input {
            for oc in 0..params.channels_output {
                for kh in 0..params.kernel_height {
                    for kw in 0..params.kernel_width {
                        let mut sum = T::default();

                        for b in 0..params.batch_size {
                            for ih in 0..params.input_height {
                                for iw in 0..params.input_width {
                                    let oh_before_pad = ih * params.stride + kh * params.dilation;
                                    let ow_before_pad = iw * params.stride + kw * params.dilation;

                                    if oh_before_pad < params.padding
                                        || oh_before_pad >= output_height + params.padding
                                        || ow_before_pad < params.padding
                                        || ow_before_pad >= output_width + params.padding
                                    {
                                        continue;
                                    }

                                    let oh = oh_before_pad - params.padding;
                                    let ow = ow_before_pad - params.padding;

                                    let input_idx = input_offset
                                        + b * input_strides[0]
                                        + ic * input_strides[1]
                                        + ih * input_strides[2]
                                        + iw * input_strides[3];
                                    let input_val = input_storage[input_idx];

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
}

/// ConvTranspose3d weight gradient
pub fn conv_transpose3d_grad_weight_map<
    T: Copy + Default + ops::Add<Output = T> + ops::Mul<Output = T> + Send + Sync,
>(
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

    let input_offset = input_layout.get_offset();
    let grad_output_offset = grad_output_layout.get_offset();
    let input_strides = input_layout.get_strides();
    let grad_output_strides = grad_output_layout.get_strides();

    #[cfg(feature = "rayon")]
    {
        let weight_grad: Vec<T> = (0..params.channels_input)
            .into_par_iter()
            .flat_map(|ic| {
                (0..params.channels_output)
                    .into_par_iter()
                    .flat_map(|oc| {
                        (0..params.kernel_depth)
                            .into_par_iter()
                            .flat_map(|kd| {
                                (0..params.kernel_height)
                                    .into_par_iter()
                                    .flat_map(|kh| {
                                        (0..params.kernel_width)
                                            .into_par_iter()
                                            .map(|kw| {
                                                let mut sum = T::default();

                                                for b in 0..params.batch_size {
                                                    for id in 0..params.input_depth {
                                                        for ih in 0..params.input_height {
                                                            for iw in 0..params.input_width {
                                                                let od_before_pad =
                                                                    id * params.stride + kd * params.dilation;
                                                                let oh_before_pad =
                                                                    ih * params.stride + kh * params.dilation;
                                                                let ow_before_pad =
                                                                    iw * params.stride + kw * params.dilation;

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

                                                                let input_idx = input_offset
                                                                    + b * input_strides[0]
                                                                    + ic * input_strides[1]
                                                                    + id * input_strides[2]
                                                                    + ih * input_strides[3]
                                                                    + iw * input_strides[4];
                                                                let input_val = input_storage[input_idx];

                                                                let grad_output_idx = grad_output_offset
                                                                    + b * grad_output_strides[0]
                                                                    + oc * grad_output_strides[1]
                                                                    + od * grad_output_strides[2]
                                                                    + oh * grad_output_strides[3]
                                                                    + ow * grad_output_strides[4];
                                                                let grad_output_val =
                                                                    grad_output_storage[grad_output_idx];

                                                                sum = sum + input_val * grad_output_val;
                                                            }
                                                        }
                                                    }
                                                }

                                                sum
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(weight_grad)
    }

    #[cfg(not(feature = "rayon"))]
    {
        let weight_grad_size = params.channels_input
            * params.channels_output
            * params.kernel_depth
            * params.kernel_height
            * params.kernel_width;
        let mut weight_grad = vec![T::default(); weight_grad_size];

        for ic in 0..params.channels_input {
            for oc in 0..params.channels_output {
                for kd in 0..params.kernel_depth {
                    for kh in 0..params.kernel_height {
                        for kw in 0..params.kernel_width {
                            let mut sum = T::default();

                            for b in 0..params.batch_size {
                                for id in 0..params.input_depth {
                                    for ih in 0..params.input_height {
                                        for iw in 0..params.input_width {
                                            let od_before_pad = id * params.stride + kd * params.dilation;
                                            let oh_before_pad = ih * params.stride + kh * params.dilation;
                                            let ow_before_pad = iw * params.stride + kw * params.dilation;

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

                                            let input_idx = input_offset
                                                + b * input_strides[0]
                                                + ic * input_strides[1]
                                                + id * input_strides[2]
                                                + ih * input_strides[3]
                                                + iw * input_strides[4];
                                            let input_val = input_storage[input_idx];

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
}
