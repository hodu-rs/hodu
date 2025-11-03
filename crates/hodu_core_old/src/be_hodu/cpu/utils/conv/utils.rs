use crate::{
    compat::*,
    error::HoduResult,
    op::conv::{
        ParamsConv1D, ParamsConv2D, ParamsConv3D, ParamsConvTranspose1D, ParamsConvTranspose2D, ParamsConvTranspose3D,
    },
    types::layout::Layout,
};

/// Conv1d weight gradient
#[cfg(feature = "rayon")]
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

/// Conv1d weight gradient
#[cfg(not(feature = "rayon"))]
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

                        let input_idx =
                            input_offset + b * input_strides[0] + ic * input_strides[1] + il_actual * input_strides[2];
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

/// Conv2d weight gradient
#[cfg(feature = "rayon")]
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

/// Conv2d weight gradient
#[cfg(not(feature = "rayon"))]
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

    let weight_grad_size = params.channels_output * params.channels_input * params.kernel_height * params.kernel_width;
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

/// Conv3d weight gradient
#[cfg(feature = "rayon")]
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
                                                            let grad_output_val = grad_output_storage[grad_output_idx];

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

/// Conv3d weight gradient
#[cfg(not(feature = "rayon"))]
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

// ConvTranspose Gradient Weight operations

/// ConvTranspose1d weight gradient
#[cfg(feature = "rayon")]
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

/// ConvTranspose1d weight gradient
#[cfg(not(feature = "rayon"))]
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

                let weight_grad_idx = ic * params.channels_output * params.kernel_size + oc * params.kernel_size + k;
                weight_grad[weight_grad_idx] = sum;
            }
        }
    }

    Ok(weight_grad)
}

/// ConvTranspose2d weight gradient
#[cfg(feature = "rayon")]
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

/// ConvTranspose2d weight gradient
#[cfg(not(feature = "rayon"))]
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

    let weight_grad_size = params.channels_input * params.channels_output * params.kernel_height * params.kernel_width;
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

/// ConvTranspose3d weight gradient
#[cfg(feature = "rayon")]
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
                                                            let grad_output_val = grad_output_storage[grad_output_idx];

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

/// ConvTranspose3d weight gradient
#[cfg(not(feature = "rayon"))]
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
