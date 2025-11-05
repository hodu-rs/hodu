use crate::{
    error::{HoduError, HoduResult},
    ops::{Op, ReduceOp},
    script::builder::ir::Attribute,
};
use hodu_xla::{XlaBuilder, XlaOp};
use std::collections::HashMap;

/// Execute reduce operations
pub fn execute(
    _builder: &XlaBuilder,
    op: &Op,
    inputs: &[XlaOp],
    attributes: &HashMap<String, Attribute>,
) -> HoduResult<XlaOp> {
    match op {
        // Reduce operations
        Op::Reduce(reduce_op) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Reduce requires input".to_string()));
            }

            // Extract keep_dim from attributes
            let keep_dim = attributes
                .get("keep_dim")
                .and_then(|attr| if let Attribute::Bool(b) = attr { Some(*b) } else { None })
                .unwrap_or(false);

            // Extract dimensions from attributes
            let dims: Vec<i64> = attributes
                .get("dims")
                .and_then(|attr| {
                    if let Attribute::IntArray(arr) = attr {
                        Some(arr.iter().map(|&d| d as i64).collect())
                    } else {
                        None
                    }
                })
                .unwrap_or_default();

            match reduce_op {
                ReduceOp::Sum => if dims.is_empty() {
                    let input_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("XLA error: {:?}", e)))?;
                    let all_dims: Vec<i64> = match input_shape {
                        hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                        _ => return Err(HoduError::InternalError("Expected array shape for reduce".to_string())),
                    };
                    inputs[0].reduce_sum(&all_dims, keep_dim)
                } else {
                    inputs[0].reduce_sum(&dims, keep_dim)
                }
                .map_err(|e| HoduError::InternalError(format!("XLA reduce_sum failed: {:?}", e))),
                ReduceOp::Mean => if dims.is_empty() {
                    let input_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("XLA error: {:?}", e)))?;
                    let all_dims: Vec<i64> = match input_shape {
                        hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                        _ => return Err(HoduError::InternalError("Expected array shape for reduce".to_string())),
                    };
                    inputs[0].reduce_mean(&all_dims, keep_dim)
                } else {
                    inputs[0].reduce_mean(&dims, keep_dim)
                }
                .map_err(|e| HoduError::InternalError(format!("XLA reduce_mean failed: {:?}", e))),
                ReduceOp::Max => if dims.is_empty() {
                    let input_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("XLA error: {:?}", e)))?;
                    let all_dims: Vec<i64> = match input_shape {
                        hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                        _ => return Err(HoduError::InternalError("Expected array shape for reduce".to_string())),
                    };
                    inputs[0].reduce_max(&all_dims, keep_dim)
                } else {
                    inputs[0].reduce_max(&dims, keep_dim)
                }
                .map_err(|e| HoduError::InternalError(format!("XLA reduce_max failed: {:?}", e))),
                ReduceOp::Min => if dims.is_empty() {
                    let input_shape = inputs[0]
                        .shape()
                        .map_err(|e| HoduError::InternalError(format!("XLA error: {:?}", e)))?;
                    let all_dims: Vec<i64> = match input_shape {
                        hodu_xla::Shape::Array(array_shape) => (0..array_shape.dims().len() as i64).collect(),
                        _ => return Err(HoduError::InternalError("Expected array shape for reduce".to_string())),
                    };
                    inputs[0].reduce_min(&all_dims, keep_dim)
                } else {
                    inputs[0].reduce_min(&dims, keep_dim)
                }
                .map_err(|e| HoduError::InternalError(format!("XLA reduce_min failed: {:?}", e))),
                _ => Err(HoduError::InternalError(format!(
                    "Reduce operation {:?} not yet implemented",
                    reduce_op
                ))),
            }
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported reduce operation: {:?}",
            op
        ))),
    }
}
