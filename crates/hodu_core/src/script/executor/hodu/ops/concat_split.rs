use crate::{
    be::storage::BackendStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    script::builder::ir::Attribute,
    types::Layout,
};

/// Execute concat and split operations
pub fn execute(
    inputs: &[Arc<BackendStorage>],
    layouts: &[Layout],
    op: &Op,
    attributes: &HashMap<String, Attribute>,
) -> HoduResult<BackendStorage> {
    match op {
        Op::Concat(_) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError(
                    "Concat requires at least one input".to_string(),
                ));
            }
            let dim = attributes
                .get("dim")
                .and_then(|a| match a {
                    Attribute::U32(d) => Some(*d),
                    Attribute::Scalar(s) => Some(s.to_u32()),
                    _ => None,
                })
                .ok_or_else(|| HoduError::InternalError("Missing dim attribute".to_string()))?;
            let other_storages: Vec<&BackendStorage> = inputs[1..].iter().map(|s| s.as_ref()).collect();
            let other_layouts: Vec<&Layout> = layouts[1..].iter().collect();
            inputs[0].call_concat(&other_storages, &other_layouts, dim, op.clone())
        },

        Op::Split(_) => {
            if inputs.len() != 1 || layouts.len() != 1 {
                return Err(HoduError::InternalError(format!(
                    "Split operation requires 1 input and layout, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            let dim = attributes
                .get("dim")
                .and_then(|a| match a {
                    Attribute::U32(d) => Some(*d),
                    Attribute::Scalar(s) => Some(s.to_u32()),
                    _ => None,
                })
                .ok_or_else(|| HoduError::InternalError("Missing dim attribute".to_string()))?;
            let start = attributes
                .get("start")
                .and_then(|a| match a {
                    Attribute::U32(s) => Some(*s),
                    Attribute::Scalar(s) => Some(s.to_u32()),
                    _ => None,
                })
                .ok_or_else(|| HoduError::InternalError("Missing start attribute".to_string()))?;
            let size = attributes
                .get("size")
                .and_then(|a| match a {
                    Attribute::U32(s) => Some(*s),
                    Attribute::Scalar(s) => Some(s.to_u32()),
                    _ => None,
                })
                .ok_or_else(|| HoduError::InternalError("Missing size attribute".to_string()))?;
            inputs[0].call_split(&layouts[0], dim, start, size, op.clone())
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported concat/split operation: {:?}",
            op
        ))),
    }
}
