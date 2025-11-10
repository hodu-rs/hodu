use crate::{
    be::storage::BackendStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    script::builder::ir::Attribute,
    types::Layout,
};

/// Execute reduce operations
pub fn execute(
    inputs: &[Arc<BackendStorage>],
    layouts: &[Layout],
    op: &Op,
    attributes: &HashMap<String, Attribute>,
) -> HoduResult<BackendStorage> {
    match op {
        Op::Reduce(_) => {
            if inputs.len() != 1 || layouts.len() != 1 {
                return Err(HoduError::InternalError(format!(
                    "Reduce operation requires 1 input and layout, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            let dims: Vec<u32> = attributes
                .get("dims")
                .and_then(|a| {
                    if let Attribute::Scalars(s) = a {
                        Some(s.iter().map(|sc| sc.to_u32()).collect())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::MissingAttribute("dims".to_string()))?;
            let keep_dim = attributes
                .get("keep_dim")
                .and_then(|a| if let Attribute::Bool(b) = a { Some(*b) } else { None })
                .unwrap_or(false);
            inputs[0].call_ops_reduce(&layouts[0], &dims, keep_dim, op.clone())
        },

        Op::Windowing(_) => {
            if inputs.len() != 1 || layouts.len() != 1 {
                return Err(HoduError::InternalError(format!(
                    "Windowing operation requires 1 input and layout, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            let window_shape: Vec<u32> = attributes
                .get("window_shape")
                .and_then(|a| {
                    if let Attribute::Scalars(s) = a {
                        Some(s.iter().map(|sc| sc.to_u32()).collect())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::MissingAttribute("window_shape".to_string()))?;
            let strides: Vec<u32> = attributes
                .get("strides")
                .and_then(|a| {
                    if let Attribute::Scalars(s) = a {
                        Some(s.iter().map(|sc| sc.to_u32()).collect())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::MissingAttribute("strides".to_string()))?;
            let padding: Vec<u32> = attributes
                .get("padding")
                .and_then(|a| {
                    if let Attribute::Scalars(s) = a {
                        Some(s.iter().map(|sc| sc.to_u32()).collect())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::MissingAttribute("padding".to_string()))?;
            inputs[0].call_ops_reduce_window(&layouts[0], &window_shape, &strides, &padding, op.clone())
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported reduce operation: {:?}",
            op
        ))),
    }
}
