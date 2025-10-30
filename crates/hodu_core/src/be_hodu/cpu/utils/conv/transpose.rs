use crate::{
    compat::*,
    error::HoduResult,
    op::conv::{ParamsConvTranspose1D, ParamsConvTranspose2D, ParamsConvTranspose3D},
    types::layout::Layout,
};

// ConvTranspose1D - Parallelize at batch level due to output accumulation
#[cfg(feature = "rayon")]
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

    let batch_results: Vec<Vec<T>> = (0..params.batch_size)
        .into_par_iter()
        .map(|b| {
            let mut batch_result = vec![T::default(); params.channels_output * output_length];

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

                        let output_idx = b * params.channels_output * output_length + oc * output_length + ol_actual;
                        result[output_idx] = result[output_idx] + input_val * weight_val;
                    }
                }
            }
        }
    }

    Ok(result)
}

// ConvTranspose2D - Parallelize at batch level
#[cfg(feature = "rayon")]
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

                                    let weight_idx = weight_base_idx + kh * weight_strides[2] + kw * weight_strides[3];
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

// ConvTranspose3D - Parallelize at batch level
#[cfg(feature = "rayon")]
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

                                            let local_idx = oc * output_depth * output_height * output_width
                                                + od_actual * output_height * output_width
                                                + oh_actual * output_width
                                                + ow_actual;
                                            batch_result[local_idx] = batch_result[local_idx] + input_val * weight_val;
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
