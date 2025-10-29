use crate::{compat::*, error::HoduResult, op::conv::ParamsConv1D, types::layout::Layout};

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
