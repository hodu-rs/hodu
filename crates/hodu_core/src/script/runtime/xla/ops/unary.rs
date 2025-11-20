use crate::{
    error::{HoduError, HoduResult},
    ops::{Op, UnaryLogicalOp, UnaryOp},
    script::builder::ir::Attribute,
};
use hodu_xla::{XlaBuilder, XlaOp};
use std::collections::HashMap;

use super::super::helpers::unary_op_check;

/// Execute unary operations
pub fn execute(
    builder: &XlaBuilder,
    op: &Op,
    inputs: &[XlaOp],
    _attributes: &HashMap<String, Attribute>,
) -> HoduResult<XlaOp> {
    match op {
        // Unary operations
        Op::Unary(UnaryOp::Neg) => unary_op_check(inputs, |i| i[0].neg(), "neg"),
        Op::Unary(UnaryOp::Abs) => unary_op_check(inputs, |i| i[0].abs(), "abs"),
        Op::Unary(UnaryOp::Sign) => unary_op_check(inputs, |i| i[0].sign(), "sign"),
        Op::Unary(UnaryOp::Square) => unary_op_check(inputs, |i| i[0].mul_(&i[0]), "square"),
        Op::Unary(UnaryOp::Sqrt) => unary_op_check(inputs, |i| i[0].sqrt(), "sqrt"),
        Op::Unary(UnaryOp::Recip) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Recip requires 1 input".to_string()));
            }
            let one = builder
                .constant_r0(1.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create one: {:?}", e)))?;
            Ok(one.div_(&inputs[0])?)
        },
        Op::Unary(UnaryOp::Relu) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Relu requires 1 input".to_string()));
            }
            let zero = builder
                .constant_r0(0.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create zero: {:?}", e)))?;
            Ok(inputs[0].max(&zero)?)
        },
        Op::Unary(UnaryOp::Sigmoid) => unary_op_check(inputs, |i| i[0].logistic(), "sigmoid"),
        Op::Unary(UnaryOp::Tanh) => unary_op_check(inputs, |i| i[0].tanh(), "tanh"),
        Op::Unary(UnaryOp::Sin) => unary_op_check(inputs, |i| i[0].sin(), "sin"),
        Op::Unary(UnaryOp::Cos) => unary_op_check(inputs, |i| i[0].cos(), "cos"),
        Op::Unary(UnaryOp::Tan) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Tan requires 1 input".to_string()));
            }
            let sin_val = inputs[0].sin()?;
            let cos_val = inputs[0].cos()?;
            Ok(sin_val.div_(&cos_val)?)
        },
        Op::Unary(UnaryOp::Exp) => unary_op_check(inputs, |i| i[0].exp(), "exp"),
        Op::Unary(UnaryOp::Exp2) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Exp2 requires 1 input".to_string()));
            }
            let ln2 = builder
                .constant_r0(2.0f32.ln())
                .map_err(|e| HoduError::InternalError(format!("Failed to create ln2: {:?}", e)))?;
            let scaled = inputs[0].mul_(&ln2)?;
            Ok(scaled.exp()?)
        },
        Op::Unary(UnaryOp::Exp10) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Exp10 requires 1 input".to_string()));
            }
            let ln10 = builder
                .constant_r0(10.0f32.ln())
                .map_err(|e| HoduError::InternalError(format!("Failed to create ln10: {:?}", e)))?;
            let scaled = inputs[0].mul_(&ln10)?;
            Ok(scaled.exp()?)
        },
        Op::Unary(UnaryOp::Ln) => unary_op_check(inputs, |i| i[0].log(), "ln"),
        Op::Unary(UnaryOp::Log2) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Log2 requires 1 input".to_string()));
            }
            let ln_val = inputs[0].log()?;
            let ln2 = builder
                .constant_r0(2.0f32.ln())
                .map_err(|e| HoduError::InternalError(format!("Failed to create ln2: {:?}", e)))?;
            Ok(ln_val.div_(&ln2)?)
        },
        Op::Unary(UnaryOp::Log10) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Log10 requires 1 input".to_string()));
            }
            let ln_val = inputs[0].log()?;
            let ln10 = builder
                .constant_r0(10.0f32.ln())
                .map_err(|e| HoduError::InternalError(format!("Failed to create ln10: {:?}", e)))?;
            Ok(ln_val.div_(&ln10)?)
        },
        Op::Unary(UnaryOp::Gelu) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Gelu requires 1 input".to_string()));
            }
            // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            let half = builder
                .constant_r0(0.5f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create constant: {:?}", e)))?;
            let one = builder
                .constant_r0(1.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create constant: {:?}", e)))?;
            let sqrt_2_pi = builder
                .constant_r0((2.0f32 / std::f32::consts::PI).sqrt())
                .map_err(|e| HoduError::InternalError(format!("Failed to create constant: {:?}", e)))?;
            let c = builder
                .constant_r0(0.044715f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create constant: {:?}", e)))?;

            let x3 = inputs[0]
                .mul_(&inputs[0])
                .and_then(|x2| x2.mul_(&inputs[0]))
                .map_err(|e| HoduError::InternalError(format!("Failed: {:?}", e)))?;
            let cx3 = c
                .mul_(&x3)
                .map_err(|e| HoduError::InternalError(format!("Failed: {:?}", e)))?;
            let inner = inputs[0]
                .add_(&cx3)
                .map_err(|e| HoduError::InternalError(format!("Failed: {:?}", e)))?;
            let scaled = sqrt_2_pi
                .mul_(&inner)
                .map_err(|e| HoduError::InternalError(format!("Failed: {:?}", e)))?;
            let tanh_val = scaled
                .tanh()
                .map_err(|e| HoduError::InternalError(format!("Failed: {:?}", e)))?;
            let one_plus = one
                .add_(&tanh_val)
                .map_err(|e| HoduError::InternalError(format!("Failed: {:?}", e)))?;
            let x_mul = inputs[0]
                .mul_(&one_plus)
                .map_err(|e| HoduError::InternalError(format!("Failed: {:?}", e)))?;
            Ok(half.mul_(&x_mul)?)
        },
        Op::Unary(UnaryOp::Softplus) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Softplus requires 1 input".to_string()));
            }
            // softplus(x) = ln(1 + exp(x))
            let one = builder
                .constant_r0(1.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create one: {:?}", e)))?;
            let exp_x = inputs[0].exp()?;
            let one_plus_exp = one.add_(&exp_x)?;
            Ok(one_plus_exp.log()?)
        },
        Op::Unary(UnaryOp::Silu) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Silu requires 1 input".to_string()));
            }
            // silu(x) = x * sigmoid(x)
            let sigmoid_x = inputs[0].logistic()?;
            Ok(inputs[0].mul_(&sigmoid_x)?)
        },
        Op::Unary(UnaryOp::Mish) => {
            if inputs.len() != 1 {
                return Err(HoduError::InternalError("Mish requires 1 input".to_string()));
            }
            // mish(x) = x * tanh(softplus(x))
            let one = builder
                .constant_r0(1.0f32)
                .map_err(|e| HoduError::InternalError(format!("Failed to create one: {:?}", e)))?;
            let exp_x = inputs[0].exp()?;
            let one_plus_exp = one.add_(&exp_x)?;
            let ln_val = one_plus_exp.log()?;
            let tanh_val = ln_val.tanh()?;
            Ok(inputs[0].mul_(&tanh_val)?)
        },

        // UnaryLogical operations
        Op::UnaryLogical(UnaryLogicalOp::LogicalNot) => unary_op_check(inputs, |i| i[0].not(), "logical_not"),

        _ => Err(HoduError::InternalError(format!(
            "Unsupported unary operation: {:?}",
            op
        ))),
    }
}
