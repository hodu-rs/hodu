use crate::{
    be::storage::BackendStorage,
    compat::*,
    error::{HoduError, HoduResult},
    ops::Op,
    script::{builder::ir::Attribute, op_params::OpParams},
    types::Layout,
};

/// Execute convolution operations
pub fn execute(
    inputs: &[&Arc<BackendStorage>],
    layouts: &[&Layout],
    op: &Op,
    attributes: &HashMap<String, Attribute>,
) -> HoduResult<BackendStorage> {
    match op {
        Op::Conv(_) => {
            if inputs.len() != 2 || layouts.len() != 2 {
                return Err(HoduError::InternalError(format!(
                    "Conv operation requires 2 inputs and layouts, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            let params = OpParams::new(attributes);
            let stride = params.get_usize_array("stride")?;
            let padding = params.get_usize_array("padding")?;
            let dilation = params.get_usize_array("dilation")?;
            inputs[0].call_ops_conv(
                layouts[0],
                inputs[1],
                layouts[1],
                &stride,
                &padding,
                &dilation,
                op.clone(),
            )
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported conv operation: {:?}",
            op
        ))),
    }
}
