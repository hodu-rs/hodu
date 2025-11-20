use crate::{
    error::{HoduError, HoduResult},
    ops::{ConvOp, Op},
    script::{builder::ir::Attribute, op_params::OpParams},
};
use hodu_xla::{XlaBuilder, XlaOp};
use std::collections::HashMap;

/// Execute convolution operations
pub fn execute(
    _builder: &XlaBuilder,
    op: &Op,
    inputs: &[XlaOp],
    attributes: &HashMap<String, Attribute>,
) -> HoduResult<XlaOp> {
    match op {
        // Conv operations
        Op::Conv(conv_op) => {
            if inputs.len() != 2 {
                return Err(HoduError::InternalError(
                    "Conv requires exactly 2 inputs (input, weight)".to_string(),
                ));
            }

            let params = OpParams::new(attributes);

            match conv_op {
                ConvOp::Conv1d => {
                    let stride = params.get_int_as_i64_opt("stride").unwrap_or(1);
                    let padding = params.get_int_as_i64_opt("padding").unwrap_or(0);
                    let dilation = params.get_int_as_i64_opt("dilation").unwrap_or(1);

                    Ok(inputs[0].conv_general_dilated(
                        &inputs[1],
                        &[stride],
                        &[(padding, padding)],
                        &[],
                        &[dilation],
                        0,
                        1,
                        &[2],
                        1,
                        0,
                        &[2],
                        1,
                        1,
                    )?)
                },
                ConvOp::Conv2d => {
                    let stride = params.get_int_as_i64_opt("stride").unwrap_or(1);
                    let padding = params.get_int_as_i64_opt("padding").unwrap_or(0);
                    let dilation = params.get_int_as_i64_opt("dilation").unwrap_or(1);

                    Ok(inputs[0].conv_general_dilated(
                        &inputs[1],
                        &[stride, stride],
                        &[(padding, padding), (padding, padding)],
                        &[],
                        &[dilation, dilation],
                        0,
                        1,
                        &[2, 3],
                        1,
                        0,
                        &[2, 3],
                        1,
                        1,
                    )?)
                },
                ConvOp::Conv3d => {
                    let stride = params.get_int_as_i64_opt("stride").unwrap_or(1);
                    let padding = params.get_int_as_i64_opt("padding").unwrap_or(0);
                    let dilation = params.get_int_as_i64_opt("dilation").unwrap_or(1);

                    Ok(inputs[0].conv_general_dilated(
                        &inputs[1],
                        &[stride, stride, stride],
                        &[(padding, padding), (padding, padding), (padding, padding)],
                        &[],
                        &[dilation, dilation, dilation],
                        0,
                        1,
                        &[2, 3, 4],
                        1,
                        0,
                        &[2, 3, 4],
                        1,
                        1,
                    )?)
                },
                ConvOp::ConvTranspose1d => {
                    let kernel_size = params.get_int_as_i64_opt("kernel_size").unwrap_or(1);
                    let padding = params.get_int_as_i64_opt("padding").unwrap_or(0);
                    let output_padding = params.get_int_as_i64_opt("output_padding").unwrap_or(0);
                    let stride = params.get_int_as_i64_opt("stride").unwrap_or(1);
                    let dilation = params.get_int_as_i64_opt("dilation").unwrap_or(1);

                    let use_lhs_dilation = stride > 1;
                    let (xla_lhs_dilation, xla_padding_low, xla_padding_high) = if use_lhs_dilation {
                        let pad_total = 2 * kernel_size - 2 - 2 * padding + output_padding;
                        let pad_low = pad_total / 2;
                        let pad_high = pad_total - pad_low;
                        (vec![stride], pad_low, pad_high)
                    } else {
                        let pad_total = dilation * kernel_size + kernel_size - 2 - 2 * padding + output_padding;
                        let pad_low = pad_total / 2;
                        let pad_high = pad_total - pad_low;
                        (vec![], pad_low, pad_high)
                    };

                    let kernel_op = if use_lhs_dilation {
                        &inputs[1]
                    } else {
                        &inputs[1].rev(&[2])?
                    };

                    Ok(inputs[0].conv_general_dilated(
                        kernel_op,
                        &[1],
                        &[(xla_padding_low, xla_padding_high)],
                        &xla_lhs_dilation,
                        &[dilation],
                        0,
                        1,
                        &[2],
                        0,
                        1,
                        &[2],
                        1,
                        1,
                    )?)
                },
                ConvOp::ConvTranspose2d => {
                    let kernel_height = params.get_int_as_i64_opt("kernel_height").unwrap_or(1);
                    let kernel_width = params.get_int_as_i64_opt("kernel_width").unwrap_or(1);
                    let padding = params.get_int_as_i64_opt("padding").unwrap_or(0);
                    let output_padding = params.get_int_as_i64_opt("output_padding").unwrap_or(0);
                    let stride = params.get_int_as_i64_opt("stride").unwrap_or(1);
                    let dilation = params.get_int_as_i64_opt("dilation").unwrap_or(1);

                    let use_lhs_dilation = stride > 1;
                    let (
                        xla_lhs_dilation,
                        xla_padding_h_low,
                        xla_padding_h_high,
                        xla_padding_w_low,
                        xla_padding_w_high,
                    ) = if use_lhs_dilation {
                        let pad_h_total = 2 * kernel_height - 2 - 2 * padding + output_padding;
                        let pad_h_low = pad_h_total / 2;
                        let pad_h_high = pad_h_total - pad_h_low;
                        let pad_w_total = 2 * kernel_width - 2 - 2 * padding + output_padding;
                        let pad_w_low = pad_w_total / 2;
                        let pad_w_high = pad_w_total - pad_w_low;
                        (vec![stride, stride], pad_h_low, pad_h_high, pad_w_low, pad_w_high)
                    } else {
                        let pad_h_total = dilation * kernel_height + kernel_height - 2 - 2 * padding + output_padding;
                        let pad_h_low = pad_h_total / 2;
                        let pad_h_high = pad_h_total - pad_h_low;
                        let pad_w_total = dilation * kernel_width + kernel_width - 2 - 2 * padding + output_padding;
                        let pad_w_low = pad_w_total / 2;
                        let pad_w_high = pad_w_total - pad_w_low;
                        (vec![], pad_h_low, pad_h_high, pad_w_low, pad_w_high)
                    };

                    let kernel_op = if use_lhs_dilation {
                        &inputs[1]
                    } else {
                        &inputs[1].rev(&[2, 3])?
                    };

                    Ok(inputs[0].conv_general_dilated(
                        kernel_op,
                        &[1, 1],
                        &[
                            (xla_padding_h_low, xla_padding_h_high),
                            (xla_padding_w_low, xla_padding_w_high),
                        ],
                        &xla_lhs_dilation,
                        &[dilation, dilation],
                        0,
                        1,
                        &[2, 3],
                        0,
                        1,
                        &[2, 3],
                        1,
                        1,
                    )?)
                },
                ConvOp::ConvTranspose3d => {
                    let kernel_depth = params.get_int_as_i64_opt("kernel_depth").unwrap_or(1);
                    let kernel_height = params.get_int_as_i64_opt("kernel_height").unwrap_or(1);
                    let kernel_width = params.get_int_as_i64_opt("kernel_width").unwrap_or(1);
                    let padding = params.get_int_as_i64_opt("padding").unwrap_or(0);
                    let output_padding = params.get_int_as_i64_opt("output_padding").unwrap_or(0);
                    let stride = params.get_int_as_i64_opt("stride").unwrap_or(1);
                    let dilation = params.get_int_as_i64_opt("dilation").unwrap_or(1);

                    let use_lhs_dilation = stride > 1;
                    let (
                        xla_lhs_dilation,
                        xla_padding_d_low,
                        xla_padding_d_high,
                        xla_padding_h_low,
                        xla_padding_h_high,
                        xla_padding_w_low,
                        xla_padding_w_high,
                    ) = if use_lhs_dilation {
                        let pad_d_total = 2 * kernel_depth - 2 - 2 * padding + output_padding;
                        let pad_d_low = pad_d_total / 2;
                        let pad_d_high = pad_d_total - pad_d_low;
                        let pad_h_total = 2 * kernel_height - 2 - 2 * padding + output_padding;
                        let pad_h_low = pad_h_total / 2;
                        let pad_h_high = pad_h_total - pad_h_low;
                        let pad_w_total = 2 * kernel_width - 2 - 2 * padding + output_padding;
                        let pad_w_low = pad_w_total / 2;
                        let pad_w_high = pad_w_total - pad_w_low;
                        (
                            vec![stride, stride, stride],
                            pad_d_low,
                            pad_d_high,
                            pad_h_low,
                            pad_h_high,
                            pad_w_low,
                            pad_w_high,
                        )
                    } else {
                        let pad_d_total = dilation * kernel_depth + kernel_depth - 2 - 2 * padding + output_padding;
                        let pad_d_low = pad_d_total / 2;
                        let pad_d_high = pad_d_total - pad_d_low;
                        let pad_h_total = dilation * kernel_height + kernel_height - 2 - 2 * padding + output_padding;
                        let pad_h_low = pad_h_total / 2;
                        let pad_h_high = pad_h_total - pad_h_low;
                        let pad_w_total = dilation * kernel_width + kernel_width - 2 - 2 * padding + output_padding;
                        let pad_w_low = pad_w_total / 2;
                        let pad_w_high = pad_w_total - pad_w_low;
                        (
                            vec![],
                            pad_d_low,
                            pad_d_high,
                            pad_h_low,
                            pad_h_high,
                            pad_w_low,
                            pad_w_high,
                        )
                    };

                    let kernel_op = if use_lhs_dilation {
                        &inputs[1]
                    } else {
                        &inputs[1].rev(&[2, 3, 4])?
                    };

                    Ok(inputs[0].conv_general_dilated(
                        kernel_op,
                        &[1, 1, 1],
                        &[
                            (xla_padding_d_low, xla_padding_d_high),
                            (xla_padding_h_low, xla_padding_h_high),
                            (xla_padding_w_low, xla_padding_w_high),
                        ],
                        &xla_lhs_dilation,
                        &[dilation, dilation, dilation],
                        0,
                        1,
                        &[2, 3, 4],
                        0,
                        1,
                        &[2, 3, 4],
                        1,
                        1,
                    )?)
                },
                _ => Err(HoduError::InternalError(format!(
                    "Conv operation {:?} not yet implemented",
                    conv_op
                ))),
            }
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported conv operation: {:?}",
            op
        ))),
    }
}
