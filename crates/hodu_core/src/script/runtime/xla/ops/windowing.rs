use crate::{
    error::{HoduError, HoduResult},
    ops::{Op, WindowingOp},
    script::{builder::ir::Attribute, op_params::OpParams},
};
use hodu_xla::{XlaBuilder, XlaOp};
use std::collections::HashMap;

/// Execute windowing operations
pub fn execute(
    builder: &XlaBuilder,
    op: &Op,
    inputs: &[XlaOp],
    attributes: &HashMap<String, Attribute>,
) -> HoduResult<XlaOp> {
    match op {
        // Windowing operations
        Op::Windowing(windowing_op) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Windowing requires input".to_string()));
            }

            let params = OpParams::new(attributes);
            let window_shape = params.get_int_array_as_usize("window_shape")?;
            let strides: Vec<usize> = params
                .get_int_array_as_usize("strides")
                .unwrap_or_else(|_| vec![1; window_shape.len()]);
            let padding_flat: Vec<i64> = params
                .get_int_array_as_i64_opt("padding")
                .unwrap_or_else(|| vec![0; window_shape.len() * 2]);

            let mut padding = Vec::with_capacity(window_shape.len());
            for i in 0..window_shape.len() {
                let pad_lo = padding_flat[i * 2] as usize;
                let pad_hi = padding_flat[i * 2 + 1] as usize;
                padding.push((pad_lo, pad_hi));
            }

            // Get element type from shape
            let shape = inputs[0]
                .shape()
                .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
            let element_type = match shape {
                hodu_xla::Shape::Array(array_shape) => array_shape.element_type(),
                _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
            };

            let input = &inputs[0];

            // Create initial value and reduction computation based on windowing operation
            let (init_value, reduction_comp, is_mean) = match windowing_op {
                WindowingOp::ReduceWindowMax => {
                    let init = builder
                        .min_value(element_type)
                        .map_err(|e| HoduError::InternalError(format!("Failed to create min_value: {:?}", e)))?;
                    let max_builder = hodu_xla::XlaBuilder::new("Max");
                    let x = max_builder
                        .parameter(0, element_type, &[], "x")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let y = max_builder
                        .parameter(1, element_type, &[], "y")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let comp = x
                        .max(&y)
                        .map_err(|e| HoduError::InternalError(format!("Failed to max: {:?}", e)))?
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build: {:?}", e)))?;
                    (init, comp, false)
                },
                WindowingOp::ReduceWindowMin => {
                    let init = builder
                        .max_value(element_type)
                        .map_err(|e| HoduError::InternalError(format!("Failed to create max_value: {:?}", e)))?;
                    let min_builder = hodu_xla::XlaBuilder::new("Min");
                    let x = min_builder
                        .parameter(0, element_type, &[], "x")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let y = min_builder
                        .parameter(1, element_type, &[], "y")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let comp = x
                        .min(&y)
                        .map_err(|e| HoduError::InternalError(format!("Failed to min: {:?}", e)))?
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build: {:?}", e)))?;
                    (init, comp, false)
                },
                WindowingOp::ReduceWindowSum | WindowingOp::ReduceWindowMean => {
                    let init = builder
                        .zero(element_type)
                        .map_err(|e| HoduError::InternalError(format!("Failed to create zero: {:?}", e)))?;
                    let add_builder = hodu_xla::XlaBuilder::new("Add");
                    let x = add_builder
                        .parameter(0, element_type, &[], "x")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let y = add_builder
                        .parameter(1, element_type, &[], "y")
                        .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;
                    let comp = x
                        .add_(&y)
                        .map_err(|e| HoduError::InternalError(format!("Failed to add: {:?}", e)))?
                        .build()
                        .map_err(|e| HoduError::InternalError(format!("Failed to build: {:?}", e)))?;
                    let is_mean = matches!(windowing_op, WindowingOp::ReduceWindowMean);
                    (init, comp, is_mean)
                },
            };

            // Convert to i64
            let window_shape_i64: Vec<i64> = window_shape.iter().map(|&v| v as i64).collect();
            let strides_i64: Vec<i64> = strides.iter().map(|&v| v as i64).collect();
            let padding_i64: Vec<(i64, i64)> = padding.iter().map(|&(lo, hi)| (lo as i64, hi as i64)).collect();

            // Apply reduce_window
            let result = input.reduce_window(
                init_value,
                reduction_comp,
                &window_shape_i64,
                &strides_i64,
                &padding_i64,
            )?;

            // For mean, divide by window size
            if is_mean {
                let window_size: usize = window_shape.iter().product();
                let window_size_scalar = builder
                    .constant_r0(window_size as f32)
                    .map_err(|e| HoduError::InternalError(format!("Failed to create constant: {:?}", e)))?;
                Ok(result.div_(&window_size_scalar)?)
            } else {
                Ok(result)
            }
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported windowing operation: {:?}",
            op
        ))),
    }
}
