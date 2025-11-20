use crate::{
    be::storage::BackendStorage,
    compat::*,
    error::{HoduError, HoduResult},
    ops::Op,
    script::builder::ir::Attribute,
    types::Layout,
};

/// Execute unary operations: Unary, UnaryLogical, UnaryScalar
pub fn execute(
    inputs: &[&Arc<BackendStorage>],
    layouts: &[&Layout],
    op: &Op,
    attributes: &HashMap<String, Attribute>,
) -> HoduResult<BackendStorage> {
    match op {
        Op::Unary(_) => {
            if inputs.len() != 1 || layouts.len() != 1 {
                return Err(HoduError::InternalError(format!(
                    "Unary operation requires 1 input and layout, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            inputs[0].call_ops_unary(layouts[0], op.clone())
        },

        Op::UnaryLogical(_) => {
            if inputs.len() != 1 || layouts.len() != 1 {
                return Err(HoduError::InternalError(format!(
                    "UnaryLogical operation requires 1 input and layout, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            inputs[0].call_ops_unary_logical(layouts[0], op.clone())
        },

        Op::UnaryScalar(_) => {
            if inputs.len() != 1 || layouts.len() != 1 {
                return Err(HoduError::InternalError(format!(
                    "UnaryScalar operation requires 1 input and layout, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            let scalar = attributes
                .get("scalar")
                .and_then(|a| if let Attribute::Scalar(s) = a { Some(*s) } else { None })
                .ok_or_else(|| HoduError::MissingAttribute("scalar".to_string()))?;
            inputs[0].call_ops_unary_scalar(layouts[0], scalar, op.clone())
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported unary operation: {:?}",
            op
        ))),
    }
}
