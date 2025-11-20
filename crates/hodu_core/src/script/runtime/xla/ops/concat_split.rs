use crate::{
    error::{HoduError, HoduResult},
    ops::Op,
    script::{builder::ir::Attribute, op_params::OpParams},
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

            let params = OpParams::new(attributes);
            let dim = params.get_int_as_i64("dim")?;

            Ok(inputs[0].concat_in_dim(&inputs[1..], dim)?)
        },

        // Split operations
        Op::Split(_split_op) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Split requires exactly 1 input".to_string()));
            }

            let params = OpParams::new(attributes);
            let dim = params.get_int_as_i64("dim")?;
            let sizes = params.get_int_array_as_i64("sizes")?;
            let output_index = params.get_usize("output_index")?;

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
            Ok(inputs[0].slice_in_dim(start_offset, start_offset + size, 1, dim)?)
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported concat/split operation: {:?}",
            op
        ))),
    }
}
