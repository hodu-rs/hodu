use crate::{
    error::{HoduError, HoduResult},
    ops::Op,
    script::builder::ir::Attribute,
};
use hodu_xla::{XlaBuilder, XlaOp};
use std::collections::HashMap;

/// Execute concat and split operations
pub fn execute(
    _builder: &XlaBuilder,
    op: &Op,
    inputs: &[XlaOp],
    attributes: &HashMap<String, Attribute>,
) -> HoduResult<XlaOp> {
    match op {
        // Concat operations
        Op::Concat(_concat_op) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Concat requires at least 1 input".to_string()));
            }

            // Extract dimension from attributes
            let dim = attributes
                .get("dim")
                .and_then(|attr| {
                    if let Attribute::Int(i) = attr {
                        Some(*i as i64)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::InternalError("Concat requires dim attribute".to_string()))?;

            inputs[0]
                .concat_in_dim(&inputs[1..], dim)
                .map_err(|e| HoduError::InternalError(format!("XLA concat failed: {:?}", e)))
        },

        // Split operations
        Op::Split(_split_op) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Split requires exactly 1 input".to_string()));
            }

            // Extract dimension from attributes
            let dim = attributes
                .get("dim")
                .and_then(|attr| {
                    if let Attribute::Int(i) = attr {
                        Some(*i as i64)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::InternalError("Split requires dim attribute".to_string()))?;

            // Extract sizes from attributes
            let sizes: Vec<i64> = attributes
                .get("sizes")
                .and_then(|attr| {
                    if let Attribute::IntArray(arr) = attr {
                        Some(arr.iter().map(|&d| d as i64).collect())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::InternalError("Split requires sizes attribute".to_string()))?;

            // Extract output_index from attributes
            let output_index = attributes
                .get("output_index")
                .and_then(|attr| {
                    if let Attribute::Int(i) = attr {
                        Some(*i as usize)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::InternalError("Split requires output_index attribute".to_string()))?;

            // Calculate split indices (cumulative sum)
            let mut split_indices = Vec::with_capacity(sizes.len() - 1);
            let mut cumsum = 0i64;
            for &size in &sizes[..sizes.len() - 1] {
                cumsum += size;
                split_indices.push(cumsum);
            }

            // Calculate start and limit for this output slice
            let start_offset = if output_index == 0 {
                0
            } else {
                split_indices[output_index - 1]
            };

            let size = sizes[output_index];

            // Use slice_in_dim operation
            inputs[0]
                .slice_in_dim(start_offset, start_offset + size, 1, dim)
                .map_err(|e| HoduError::InternalError(format!("XLA slice_in_dim failed: {:?}", e)))
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported concat/split operation: {:?}",
            op
        ))),
    }
}
