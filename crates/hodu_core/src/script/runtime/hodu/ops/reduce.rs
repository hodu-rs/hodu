use crate::{
    be::storage::BackendStorage,
    compat::*,
    error::{HoduError, HoduResult},
    ops::Op,
    script::{builder::ir::Attribute, op_params::OpParams},
    types::Layout,
};

/// Execute reduce operations
pub fn execute(
    inputs: &[&Arc<BackendStorage>],
    layouts: &[&Layout],
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
            let params = OpParams::new(attributes);
            let dims = params.get_usize_array("dims")?;
            let keep_dim = params.get_bool_opt("keep_dim").unwrap_or(false);
            inputs[0].call_ops_reduce(layouts[0], &dims, keep_dim, op.clone())
        },

        Op::Windowing(_) => {
            if inputs.len() != 1 || layouts.len() != 1 {
                return Err(HoduError::InternalError(format!(
                    "Windowing operation requires 1 input and layout, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            let params = OpParams::new(attributes);
            let window_shape = params.get_usize_array("window_shape")?;
            let strides = params.get_usize_array("strides")?;
            let padding = params.get_usize_array("padding")?;
            inputs[0].call_ops_reduce_window(layouts[0], &window_shape, &strides, &padding, op.clone())
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported reduce operation: {:?}",
            op
        ))),
    }
}
