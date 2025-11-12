use crate::{
    be::storage::BackendStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    script::builder::ir::Attribute,
    types::Layout,
};

/// Execute convolution operations
pub fn execute(
    inputs: &[&Arc<BackendStorage>],
    layouts: &[Layout],
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
            let stride: Vec<u32> = attributes
                .get("stride")
                .and_then(|a| {
                    if let Attribute::Scalars(s) = a {
                        Some(s.iter().map(|sc| sc.to_u32()).collect())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::MissingAttribute("stride".to_string()))?;
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
            let dilation: Vec<u32> = attributes
                .get("dilation")
                .and_then(|a| {
                    if let Attribute::Scalars(s) = a {
                        Some(s.iter().map(|sc| sc.to_u32()).collect())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::MissingAttribute("dilation".to_string()))?;
            inputs[0].call_ops_conv(
                &layouts[0],
                inputs[1],
                &layouts[1],
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
