use crate::{
    be::storage::BackendStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{CastOp, MemoryOp, Op},
    script::builder::ir::Attribute,
    types::Layout,
};

/// Execute shape, cast, and memory operations
pub fn execute(
    inputs: &[&Arc<BackendStorage>],
    layouts: &[&Layout],
    op: &Op,
    attributes: &HashMap<String, Attribute>,
) -> HoduResult<BackendStorage> {
    match op {
        Op::Shape(_) | Op::ShapeScalars(_) => {
            // Shape operations don't modify storage, just return input storage
            if inputs.len() != 1 {
                return Err(HoduError::InternalError(format!(
                    "Shape operation requires 1 input, got {}",
                    inputs.len()
                )));
            }
            Ok(inputs[0].as_ref().clone())
        },

        Op::Cast(CastOp::ToDType) => {
            if inputs.len() != 1 || layouts.len() != 1 {
                return Err(HoduError::InternalError(format!(
                    "Cast operation requires 1 input and layout, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            let target_dtype = attributes
                .get("dtype")
                .and_then(|a| if let Attribute::DType(dt) = a { Some(*dt) } else { None })
                .ok_or_else(|| HoduError::MissingAttribute("dtype".to_string()))?;
            inputs[0].to_dtype(layouts[0], target_dtype)
        },

        Op::Memory(MemoryOp::Contiguous) => {
            if inputs.len() != 1 || layouts.len() != 1 {
                return Err(HoduError::InternalError(format!(
                    "Contiguous operation requires 1 input and layout, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            inputs[0].contiguous(layouts[0])
        },

        Op::Dummy => {
            // Dummy op just returns the first input
            if let Some(input) = inputs.first() {
                Ok(input.as_ref().clone())
            } else {
                Err(HoduError::InternalError(
                    "Dummy operation requires one input".to_string(),
                ))
            }
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported shape/cast/memory operation: {:?}",
            op
        ))),
    }
}
