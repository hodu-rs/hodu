use crate::{
    error::{HoduError, HoduResult},
    ops::{Op, UnaryScalarOp},
    script::builder::ir::Attribute,
};
use hodu_xla::{XlaBuilder, XlaOp};
use std::collections::HashMap;

use super::super::helpers::{get_scalar_from_attributes, scalar_op};

/// Execute unary scalar operations
pub fn execute(
    builder: &XlaBuilder,
    op: &Op,
    inputs: &[XlaOp],
    attributes: &HashMap<String, Attribute>,
) -> HoduResult<XlaOp> {
    match op {
        // UnaryScalar operations
        Op::UnaryScalar(UnaryScalarOp::AddScalar) => {
            scalar_op(builder, inputs, attributes, |i, s| i[0].add_(&s), "add_scalar")
        },
        Op::UnaryScalar(UnaryScalarOp::SubScalar) => {
            scalar_op(builder, inputs, attributes, |i, s| i[0].sub_(&s), "sub_scalar")
        },
        Op::UnaryScalar(UnaryScalarOp::MulScalar) => {
            scalar_op(builder, inputs, attributes, |i, s| i[0].mul_(&s), "mul_scalar")
        },
        Op::UnaryScalar(UnaryScalarOp::DivScalar) => {
            scalar_op(builder, inputs, attributes, |i, s| i[0].div_(&s), "div_scalar")
        },
        Op::UnaryScalar(UnaryScalarOp::PowScalar) => {
            scalar_op(builder, inputs, attributes, |i, s| i[0].pow(&s), "pow_scalar")
        },
        Op::UnaryScalar(UnaryScalarOp::MaximumScalar) => {
            scalar_op(builder, inputs, attributes, |i, s| i[0].max(&s), "maximum_scalar")
        },
        Op::UnaryScalar(UnaryScalarOp::MinimumScalar) => {
            scalar_op(builder, inputs, attributes, |i, s| i[0].min(&s), "minimum_scalar")
        },
        Op::UnaryScalar(UnaryScalarOp::LeakyRelu) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("LeakyRelu requires 1 input".to_string()));
            }
            let alpha = get_scalar_from_attributes(attributes)?;
            let zero = builder
                .constant_r0(0.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create zero: {:?}", e)))?;
            let alpha_op = builder
                .constant_r0(alpha)
                .map_err(|e| HoduError::InternalError(format!("Failed to create alpha: {:?}", e)))?;
            let cond = inputs[0]
                .gt(&zero)
                .map_err(|e| HoduError::InternalError(format!("XLA gt failed: {:?}", e)))?;
            let neg_part = inputs[0]
                .mul_(&alpha_op)
                .map_err(|e| HoduError::InternalError(format!("XLA mul failed: {:?}", e)))?;
            cond.select(&inputs[0], &neg_part)
                .map_err(|e| HoduError::InternalError(format!("XLA leaky_relu failed: {:?}", e)))
        },
        Op::UnaryScalar(UnaryScalarOp::Elu) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Elu requires 1 input".to_string()));
            }
            let alpha = get_scalar_from_attributes(attributes)?;
            let zero = builder
                .constant_r0(0.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create zero: {:?}", e)))?;
            let alpha_op = builder
                .constant_r0(alpha)
                .map_err(|e| HoduError::InternalError(format!("Failed to create alpha: {:?}", e)))?;
            let one = builder
                .constant_r0(1.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create one: {:?}", e)))?;
            let cond = inputs[0]
                .gt(&zero)
                .map_err(|e| HoduError::InternalError(format!("XLA gt failed: {:?}", e)))?;
            let exp_x = inputs[0]
                .exp()
                .map_err(|e| HoduError::InternalError(format!("XLA exp failed: {:?}", e)))?;
            let exp_minus_one = exp_x
                .sub_(&one)
                .map_err(|e| HoduError::InternalError(format!("XLA sub failed: {:?}", e)))?;
            let neg_part = alpha_op
                .mul_(&exp_minus_one)
                .map_err(|e| HoduError::InternalError(format!("XLA mul failed: {:?}", e)))?;
            cond.select(&inputs[0], &neg_part)
                .map_err(|e| HoduError::InternalError(format!("XLA elu failed: {:?}", e)))
        },
        Op::UnaryScalar(UnaryScalarOp::Prelu) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Prelu requires 1 input".to_string()));
            }
            let alpha = get_scalar_from_attributes(attributes)?;
            let zero = builder
                .constant_r0(0.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create zero: {:?}", e)))?;
            let alpha_op = builder
                .constant_r0(alpha)
                .map_err(|e| HoduError::InternalError(format!("Failed to create alpha: {:?}", e)))?;
            let cond = inputs[0]
                .gt(&zero)
                .map_err(|e| HoduError::InternalError(format!("XLA gt failed: {:?}", e)))?;
            let neg_part = inputs[0]
                .mul_(&alpha_op)
                .map_err(|e| HoduError::InternalError(format!("XLA mul failed: {:?}", e)))?;
            cond.select(&inputs[0], &neg_part)
                .map_err(|e| HoduError::InternalError(format!("XLA prelu failed: {:?}", e)))
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported unary scalar operation: {:?}",
            op
        ))),
    }
}
