pub mod binary;
pub mod cast_memory;
pub mod concat_split;
pub mod conv;
pub mod indexing;
pub mod matrix;
pub mod reduce;
pub mod shape;
pub mod unary;
pub mod unary_scalar;
pub mod windowing;

use crate::{
    error::{HoduError, HoduResult},
    ops::Op,
    script::builder::ir::Attribute,
    script::compiler::{CompiledInstruction, CompiledModule},
};
use hodu_xla::{XlaBuilder, XlaOp};
use std::collections::HashMap;

/// Main dispatcher for XLA operations
pub fn execute_xla_op(
    builder: &XlaBuilder,
    op: &Op,
    inputs: &[XlaOp],
    attributes: &HashMap<String, Attribute>,
    compiled: &CompiledModule,
    instr: &CompiledInstruction,
) -> HoduResult<XlaOp> {
    match op {
        // Binary, BinaryLogical, Cmp, and CmpScalar operations
        Op::Binary(..) | Op::BinaryLogical(..) | Op::Cmp(..) | Op::CmpScalar(..) => {
            binary::execute(builder, op, inputs, attributes)
        },

        // Unary and UnaryLogical operations
        Op::Unary(..) | Op::UnaryLogical(..) => unary::execute(builder, op, inputs, attributes),

        // UnaryScalar operations
        Op::UnaryScalar(..) => unary_scalar::execute(builder, op, inputs, attributes),

        // Matrix operations
        Op::Matrix(..) => matrix::execute(builder, op, inputs, attributes),

        // Reduce operations
        Op::Reduce(..) => reduce::execute(builder, op, inputs, attributes),

        // Concat and Split operations
        Op::Concat(..) | Op::Split(..) => concat_split::execute(builder, op, inputs, attributes),

        // Indexing operations
        Op::Indexing(..) => indexing::execute(builder, op, inputs, attributes),

        // Conv operations
        Op::Conv(..) => conv::execute(builder, op, inputs, attributes),

        // Windowing operations
        Op::Windowing(..) => windowing::execute(builder, op, inputs, attributes),

        // Shape and ShapeScalars operations
        Op::Shape(..) | Op::ShapeScalars(..) => shape::execute(builder, op, inputs, attributes, compiled, instr.result),

        // Cast and Memory operations
        Op::Cast(..) | Op::Memory(..) => cast_memory::execute(builder, op, inputs, attributes, compiled, instr.result),

        // Dummy operation
        Op::Dummy => {
            if inputs.is_empty() {
                builder
                    .constant_r0(0.0f32)
                    .map_err(|e| HoduError::InternalError(format!("Failed to create dummy op: {:?}", e)))
            } else {
                Ok(inputs[0].clone())
            }
        },
    }
}
