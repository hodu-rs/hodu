use crate::{
    error::{HoduError, HoduResult},
    ops::{BinaryLogicalOp, BinaryOp, CmpOp, CmpScalarOp, Op},
    script::builder::ir::Attribute,
};
use hodu_xla::{XlaBuilder, XlaOp};
use std::collections::HashMap;

use super::super::helpers::{binary_op_check, scalar_op};

/// Execute binary, binary logical, and comparison operations
pub fn execute(
    builder: &XlaBuilder,
    op: &Op,
    inputs: &[XlaOp],
    attributes: &HashMap<String, Attribute>,
) -> HoduResult<XlaOp> {
    match op {
        // Binary operations
        Op::Binary(BinaryOp::Add) => binary_op_check(inputs, 2, |i| i[0].add_(&i[1]), "add"),
        Op::Binary(BinaryOp::Sub) => binary_op_check(inputs, 2, |i| i[0].sub_(&i[1]), "sub"),
        Op::Binary(BinaryOp::Mul) => binary_op_check(inputs, 2, |i| i[0].mul_(&i[1]), "mul"),
        Op::Binary(BinaryOp::Div) => binary_op_check(inputs, 2, |i| i[0].div_(&i[1]), "div"),
        Op::Binary(BinaryOp::Pow) => binary_op_check(inputs, 2, |i| i[0].pow(&i[1]), "pow"),
        Op::Binary(BinaryOp::Maximum) => binary_op_check(inputs, 2, |i| i[0].max(&i[1]), "max"),
        Op::Binary(BinaryOp::Minimum) => binary_op_check(inputs, 2, |i| i[0].min(&i[1]), "min"),

        // BinaryLogical operations
        Op::BinaryLogical(BinaryLogicalOp::LogicalAnd) => binary_op_check(inputs, 2, |i| i[0].and(&i[1]), "and"),
        Op::BinaryLogical(BinaryLogicalOp::LogicalOr) => binary_op_check(inputs, 2, |i| i[0].or(&i[1]), "or"),
        Op::BinaryLogical(BinaryLogicalOp::LogicalXor) => binary_op_check(inputs, 2, |i| i[0].xor(&i[1]), "xor"),

        // Cmp operations
        Op::Cmp(CmpOp::Eq) => binary_op_check(inputs, 2, |i| i[0].eq(&i[1]), "eq"),
        Op::Cmp(CmpOp::Ne) => binary_op_check(inputs, 2, |i| i[0].ne(&i[1]), "ne"),
        Op::Cmp(CmpOp::Lt) => binary_op_check(inputs, 2, |i| i[0].lt(&i[1]), "lt"),
        Op::Cmp(CmpOp::Le) => binary_op_check(inputs, 2, |i| i[0].le(&i[1]), "le"),
        Op::Cmp(CmpOp::Gt) => binary_op_check(inputs, 2, |i| i[0].gt(&i[1]), "gt"),
        Op::Cmp(CmpOp::Ge) => binary_op_check(inputs, 2, |i| i[0].ge(&i[1]), "ge"),

        // CmpScalar operations
        Op::CmpScalar(CmpScalarOp::EqScalar) => scalar_op(builder, inputs, attributes, |i, s| i[0].eq(&s), "eq_scalar"),
        Op::CmpScalar(CmpScalarOp::NeScalar) => scalar_op(builder, inputs, attributes, |i, s| i[0].ne(&s), "ne_scalar"),
        Op::CmpScalar(CmpScalarOp::LtScalar) => scalar_op(builder, inputs, attributes, |i, s| i[0].lt(&s), "lt_scalar"),
        Op::CmpScalar(CmpScalarOp::LeScalar) => scalar_op(builder, inputs, attributes, |i, s| i[0].le(&s), "le_scalar"),
        Op::CmpScalar(CmpScalarOp::GtScalar) => scalar_op(builder, inputs, attributes, |i, s| i[0].gt(&s), "gt_scalar"),
        Op::CmpScalar(CmpScalarOp::GeScalar) => scalar_op(builder, inputs, attributes, |i, s| i[0].ge(&s), "ge_scalar"),

        _ => Err(HoduError::InternalError(format!(
            "Unsupported binary operation: {:?}",
            op
        ))),
    }
}
