use crate::{
    error::{HoduError, HoduResult},
    ops::{MatrixOp, Op},
    script::builder::ir::Attribute,
};
use hodu_xla::{XlaBuilder, XlaOp};
use std::collections::HashMap;

use super::super::helpers::binary_op_check;

/// Execute matrix operations
pub fn execute(
    _builder: &XlaBuilder,
    op: &Op,
    inputs: &[XlaOp],
    _attributes: &HashMap<String, Attribute>,
) -> HoduResult<XlaOp> {
    match op {
        // Matrix operations
        Op::Matrix(MatrixOp::Matmul) => binary_op_check(inputs, 2, |i| i[0].dot(&i[1]), "matmul"),
        Op::Matrix(MatrixOp::Dot) => binary_op_check(inputs, 2, |i| i[0].dot(&i[1]), "dot"),

        _ => Err(HoduError::InternalError(format!(
            "Unsupported matrix operation: {:?}",
            op
        ))),
    }
}
