use crate::{
    be::storage::BackendStorage,
    compat::*,
    error::{HoduError, HoduResult},
    ops::Op,
    script::builder::ir::Attribute,
    types::Layout,
};

/// Execute binary operations: Binary, BinaryLogical, Cmp, CmpScalar
pub fn execute(
    inputs: &[&Arc<BackendStorage>],
    layouts: &[&Layout],
    op: &Op,
    attributes: &HashMap<String, Attribute>,
) -> HoduResult<BackendStorage> {
    match op {
        Op::Binary(_) => {
            if inputs.len() != 2 || layouts.len() != 2 {
                return Err(HoduError::InternalError(format!(
                    "Binary operation requires 2 inputs and layouts, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            inputs[0].call_ops_binary(inputs[1], layouts[0], layouts[1], op.clone())
        },

        Op::BinaryLogical(_) => {
            if inputs.len() != 2 || layouts.len() != 2 {
                return Err(HoduError::InternalError(format!(
                    "BinaryLogical operation requires 2 inputs and layouts, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            inputs[0].call_ops_binary_logical(inputs[1], layouts[0], layouts[1], op.clone())
        },

        Op::Cmp(_) => {
            if inputs.len() != 2 || layouts.len() != 2 {
                return Err(HoduError::InternalError(format!(
                    "Cmp operation requires 2 inputs and layouts, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            inputs[0].call_ops_cmp(inputs[1], layouts[0], layouts[1], op.clone())
        },

        Op::CmpScalar(_) => {
            if inputs.len() != 1 || layouts.len() != 1 {
                return Err(HoduError::InternalError(format!(
                    "CmpScalar operation requires 1 input and layout, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            let scalar = attributes
                .get("scalar")
                .and_then(|a| if let Attribute::Scalar(s) = a { Some(*s) } else { None })
                .ok_or_else(|| HoduError::MissingAttribute("scalar".to_string()))?;
            inputs[0].call_ops_cmp_scalar(layouts[0], scalar, op.clone())
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported binary operation: {:?}",
            op
        ))),
    }
}
