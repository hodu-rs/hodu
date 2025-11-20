use crate::{
    be::storage::BackendStorage,
    compat::*,
    error::{HoduError, HoduResult},
    ops::Op,
    script::{builder::ir::Attribute, op_params::OpParams},
    types::Layout,
};

/// Execute concat and split operations
pub fn execute(
    inputs: &[&Arc<BackendStorage>],
    layouts: &[&Layout],
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
            let params = OpParams::new(attributes);
            let dim = params.get_usize("dim")?;
            let other_storages: Vec<&BackendStorage> = inputs[1..].iter().map(|s| s.as_ref()).collect();
            let other_layouts: Vec<&Layout> = layouts[1..].to_vec();
            inputs[0].call_ops_concat(&other_storages, &other_layouts, dim, op.clone())
        },

        Op::Split(_) => {
            if inputs.len() != 1 || layouts.len() != 1 {
                return Err(HoduError::InternalError(format!(
                    "Split operation requires 1 input and layout, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            let params = OpParams::new(attributes);
            let dim = params.get_usize("dim")?;
            let start = params.get_usize("start")?;
            let size = params.get_usize("size")?;
            inputs[0].call_ops_split(layouts[0], dim, start, size, op.clone())
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported concat/split operation: {:?}",
            op
        ))),
    }
}
