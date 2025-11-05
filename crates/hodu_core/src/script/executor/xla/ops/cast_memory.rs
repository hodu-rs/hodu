use crate::{
    error::{HoduError, HoduResult},
    ops::{CastOp, MemoryOp, Op},
    script::builder::ir::{Attribute, ValueId},
    script::compiler::CompiledModule,
};
use hodu_xla::{XlaBuilder, XlaOp};
use std::collections::HashMap;

/// Execute cast and memory operations
pub fn execute(
    _builder: &XlaBuilder,
    op: &Op,
    inputs: &[XlaOp],
    _attributes: &HashMap<String, Attribute>,
    compiled: &CompiledModule,
    result_value_id: ValueId,
) -> HoduResult<XlaOp> {
    match op {
        // Cast operations
        Op::Cast(CastOp::ToDType) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("ToDType requires input".to_string()));
            }

            // Get target dtype from output value_dtypes
            let target_dtype = compiled
                .value_dtypes
                .get(&result_value_id)
                .ok_or_else(|| HoduError::InternalError("Output dtype not found".to_string()))?;

            // Convert DType to XLA PrimitiveType
            let target_element_type = match target_dtype {
                crate::types::DType::BOOL => hodu_xla::PrimitiveType::Pred,
                crate::types::DType::BF16 => hodu_xla::PrimitiveType::Bf16,
                crate::types::DType::F16 => hodu_xla::PrimitiveType::F16,
                crate::types::DType::F32 => hodu_xla::PrimitiveType::F32,
                #[cfg(feature = "f64")]
                crate::types::DType::F64 => hodu_xla::PrimitiveType::F64,
                crate::types::DType::U8 => hodu_xla::PrimitiveType::U8,
                #[cfg(feature = "u16")]
                crate::types::DType::U16 => hodu_xla::PrimitiveType::U16,
                crate::types::DType::U32 => hodu_xla::PrimitiveType::U32,
                #[cfg(feature = "u64")]
                crate::types::DType::U64 => hodu_xla::PrimitiveType::U64,
                crate::types::DType::I8 => hodu_xla::PrimitiveType::S8,
                #[cfg(feature = "i16")]
                crate::types::DType::I16 => hodu_xla::PrimitiveType::S16,
                crate::types::DType::I32 => hodu_xla::PrimitiveType::S32,
                #[cfg(feature = "i64")]
                crate::types::DType::I64 => hodu_xla::PrimitiveType::S64,
                _ => {
                    return Err(HoduError::InternalError(format!(
                        "XLA does not support dtype: {:?}",
                        target_dtype
                    )))
                },
            };

            inputs[0]
                .convert(target_element_type)
                .map_err(|e| HoduError::InternalError(format!("XLA convert failed: {:?}", e)))
        },

        // Memory operations
        Op::Memory(MemoryOp::Contiguous) => {
            if inputs.is_empty() {
                return Err(HoduError::InternalError("Contiguous requires input".to_string()));
            }
            // In XLA, all data is contiguous
            Ok(inputs[0].clone())
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported cast/memory operation: {:?}",
            op
        ))),
    }
}
