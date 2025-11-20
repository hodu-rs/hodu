use crate::{
    be::storage::BackendStorage,
    compat::*,
    error::{HoduError, HoduResult},
    ops::Op,
    script::builder::ir::Attribute,
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
            let stride: Vec<usize> = attributes
                .get("stride")
                .and_then(|a| {
                    if let Attribute::Scalars(s) = a {
                        Some(s.iter().map(|sc| sc.to_usize()).collect())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::MissingAttribute("stride".to_string()))?;
            let padding: Vec<usize> = attributes
                .get("padding")
                .and_then(|a| {
                    if let Attribute::Scalars(s) = a {
                        Some(s.iter().map(|sc| sc.to_usize()).collect())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::MissingAttribute("padding".to_string()))?;
            let dilation: Vec<usize> = attributes
                .get("dilation")
                .and_then(|a| {
                    if let Attribute::Scalars(s) = a {
                        Some(s.iter().map(|sc| sc.to_usize()).collect())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| HoduError::MissingAttribute("dilation".to_string()))?;
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
