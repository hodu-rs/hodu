use crate::{
    error::{HoduError, HoduResult},
    ops::{Op, ShapeOp, ShapeScalarsOp},
    script::builder::ir::{Attribute, ValueId},
    script::compiler::CompiledModule,
};
use hodu_xla::{XlaBuilder, XlaOp};
use std::collections::HashMap;

/// Execute shape operations
pub fn execute(
    _builder: &XlaBuilder,
    op: &Op,
    inputs: &[XlaOp],
    attributes: &HashMap<String, Attribute>,
    compiled: &CompiledModule,
    result_value_id: ValueId,
) -> HoduResult<XlaOp> {
    match op {
        // Shape operations
        Op::Shape(ShapeOp::Reshape) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Reshape requires input".to_string()));
            }

            // Get target shape from output layout
            let output_layout = compiled
                .value_layouts
                .get(&result_value_id)
                .ok_or_else(|| HoduError::InternalError("Output layout not found".to_string()))?;
            let target_shape: Vec<i64> = output_layout.shape().dims().iter().map(|&d| d as i64).collect();

            Ok(inputs[0].reshape(&target_shape)?)
        },
        Op::Shape(ShapeOp::Flatten) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Flatten requires input".to_string()));
            }
            let input_shape = inputs[0]
                .shape()
                .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
            let total_size = match input_shape {
                hodu_xla::Shape::Array(array_shape) => array_shape.dims().iter().product::<i64>(),
                _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
            };
            Ok(inputs[0].reshape(&[total_size])?)
        },
        Op::Shape(ShapeOp::Squeeze) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Squeeze requires input".to_string()));
            }
            // Get current shape and remove dimensions of size 1
            let input_shape = inputs[0]
                .shape()
                .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
            let dims = match input_shape {
                hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
            };
            let squeezed_dims: Vec<i64> = dims.iter().filter(|&&d| d != 1).copied().collect();
            if squeezed_dims.is_empty() {
                // If all dimensions are 1, keep at least one
                Ok(inputs[0].reshape(&[1])?)
            } else {
                Ok(inputs[0].reshape(&squeezed_dims)?)
            }
        },
        Op::Shape(ShapeOp::Unsqueeze) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Unsqueeze requires input".to_string()));
            }

            // Get target shape from output layout
            let output_layout = compiled
                .value_layouts
                .get(&result_value_id)
                .ok_or_else(|| HoduError::InternalError("Output layout not found".to_string()))?;
            let target_dims: Vec<i64> = output_layout.shape().dims().iter().map(|&d| d as i64).collect();

            Ok(inputs[0].reshape(&target_dims)?)
        },
        Op::Shape(ShapeOp::Broadcast) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Broadcast requires input".to_string()));
            }

            // Get target shape from output layout
            let output_layout = compiled
                .value_layouts
                .get(&result_value_id)
                .ok_or_else(|| HoduError::InternalError("Output layout not found".to_string()))?;
            let target_i64: Vec<i64> = output_layout.shape().dims().iter().map(|&d| d as i64).collect();

            let input_shape = inputs[0]
                .shape()
                .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
            let input_dims = match input_shape {
                hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
            };

            if input_dims == target_i64 {
                Ok(inputs[0].clone())
            } else {
                let input_rank = input_dims.len();
                let target_rank = target_i64.len();

                if input_rank <= target_rank {
                    let broadcast_dims: Vec<i64> = (target_rank - input_rank..target_rank).map(|i| i as i64).collect();
                    Ok(inputs[0].broadcast_in_dim(&target_i64, &broadcast_dims)?)
                } else {
                    Err(HoduError::InternalError(
                        "Cannot broadcast to smaller shape".to_string(),
                    ))
                }
            }
        },
        Op::Shape(ShapeOp::Transpose) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Transpose requires input".to_string()));
            }

            // Default transpose: swap last two dimensions
            let shape = inputs[0]
                .shape()
                .ok()
                .and_then(|s| match s {
                    hodu_xla::Shape::Array(array_shape) => {
                        let rank = array_shape.dims().len();
                        if rank < 2 {
                            None
                        } else {
                            let mut perm: Vec<i64> = (0..rank as i64).collect();
                            perm.swap(rank - 2, rank - 1);
                            Some(perm)
                        }
                    },
                    _ => None,
                })
                .ok_or_else(|| HoduError::InternalError("Failed to compute transpose permutation".to_string()))?;

            Ok(inputs[0].transpose(&shape)?)
        },
        Op::Shape(ShapeOp::Permute) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Permute requires input".to_string()));
            }

            // Get permutation from attributes
            let perm_i64: Vec<i64> = attributes
                .get("perm")
                .and_then(|attr| {
                    if let Attribute::IntArray(arr) = attr {
                        Some(arr.iter().map(|&d| d as i64).collect())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::InternalError("Permute requires perm attribute".to_string()))?;

            Ok(inputs[0].transpose(&perm_i64)?)
        },

        // ShapeScalars operations
        Op::ShapeScalars(ShapeScalarsOp::Slice) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Slice requires input".to_string()));
            }

            // Extract slice parameters from attributes
            let dim = attributes
                .get("dim")
                .and_then(|attr| {
                    if let Attribute::Int(i) = attr {
                        Some(*i as i64)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::InternalError("Slice requires dim attribute".to_string()))?;

            let start = attributes
                .get("start")
                .and_then(|attr| {
                    if let Attribute::Int(i) = attr {
                        Some(*i as i64)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::InternalError("Slice requires start attribute".to_string()))?;

            let end_value = attributes
                .get("end")
                .and_then(|attr| if let Attribute::Int(i) = attr { Some(*i) } else { None })
                .ok_or_else(|| HoduError::InternalError("Slice requires end attribute".to_string()))?;

            let stride = attributes
                .get("stride")
                .and_then(|attr| {
                    if let Attribute::Int(i) = attr {
                        Some(*i as i64)
                    } else {
                        None
                    }
                })
                .unwrap_or(1);

            // Get input shape to compute actual indices
            let input_shape = inputs[0]
                .shape()
                .map_err(|e| HoduError::InternalError(format!("Failed to get shape: {:?}", e)))?;
            let dims = match input_shape {
                hodu_xla::Shape::Array(array_shape) => array_shape.dims().to_vec(),
                _ => return Err(HoduError::InternalError("Expected array shape".to_string())),
            };

            let dim_size = dims[dim as usize];
            let start_idx = if start < 0 { dim_size + start } else { start };
            let end_idx = if end_value == i32::MAX {
                if stride > 0 {
                    dim_size
                } else {
                    -1
                }
            } else {
                let end = end_value as i64;
                if end < 0 {
                    dim_size + end
                } else {
                    end
                }
            };

            Ok(inputs[0].slice_in_dim(start_idx, end_idx, stride, dim)?)
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported shape operation: {:?}",
            op
        ))),
    }
}
