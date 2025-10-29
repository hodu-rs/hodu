use crate::{compat::*, error::HoduResult, op::conv::ParamsConv3D, types::layout::Layout};

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
