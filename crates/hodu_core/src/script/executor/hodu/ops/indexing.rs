use crate::{
    be::storage::BackendStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{IndexingOp, Op},
    script::builder::ir::Attribute,
    types::Layout,
};

/// Execute indexing operations: IndexSelect, IndexPut, Gather, Scatter
pub fn execute(
    inputs: &[&Arc<BackendStorage>],
    layouts: &[Layout],
    op: &Op,
    attributes: &HashMap<String, Attribute>,
) -> HoduResult<BackendStorage> {
    match op {
        Op::Indexing(IndexingOp::IndexSelect) => {
            if inputs.len() != 2 || layouts.len() != 2 {
                return Err(HoduError::InternalError(format!(
                    "IndexSelect requires 2 inputs and layouts, got {} and {}",
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
                .ok_or_else(|| HoduError::MissingAttribute("dim".to_string()))?;
            inputs[0].call_ops_index_select(&layouts[0], inputs[1], &layouts[1], dim, op.clone())
        },

        Op::Indexing(IndexingOp::IndexPut) => {
            if inputs.len() != 3 || layouts.len() != 3 {
                return Err(HoduError::InternalError(format!(
                    "IndexPut requires 3 inputs and layouts, got {} and {}",
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
                .ok_or_else(|| HoduError::MissingAttribute("dim".to_string()))?;
            inputs[0].call_ops_index_put(
                &layouts[0],
                inputs[1],
                &layouts[1],
                inputs[2],
                &layouts[2],
                dim,
                op.clone(),
            )
        },

        Op::Indexing(IndexingOp::Gather) => {
            if inputs.len() != 2 || layouts.len() != 2 {
                return Err(HoduError::InternalError(format!(
                    "Gather requires 2 inputs and layouts, got {} and {}",
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
                .ok_or_else(|| HoduError::MissingAttribute("dim".to_string()))?;
            inputs[0].call_ops_gather(&layouts[0], inputs[1], &layouts[1], dim, op.clone())
        },

        Op::Indexing(IndexingOp::Scatter)
        | Op::Indexing(IndexingOp::ScatterAdd)
        | Op::Indexing(IndexingOp::ScatterMax)
        | Op::Indexing(IndexingOp::ScatterMin) => {
            if inputs.len() != 3 || layouts.len() != 3 {
                return Err(HoduError::InternalError(format!(
                    "Scatter operation requires 3 inputs and layouts, got {} and {}",
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
                .ok_or_else(|| HoduError::MissingAttribute("dim".to_string()))?;
            inputs[0].call_ops_scatter(
                &layouts[0],
                inputs[1],
                &layouts[1],
                inputs[2],
                &layouts[2],
                dim,
                op.clone(),
            )
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported indexing operation: {:?}",
            op
        ))),
    }
}
