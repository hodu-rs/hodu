mod binary;
mod concat_split;
mod conv;
mod indexing;
mod matrix;
mod reduce;
mod shape_cast_memory;
mod unary;

use crate::{
    be::storage::BackendStorage, error::HoduResult, layer::compat::*, ops::Op, script::builder::ir::Attribute,
    types::Layout,
};

/// Execute a single operation by dispatching to the appropriate module
pub fn execute_operation(
    op: &Op,
    inputs: &[&Arc<BackendStorage>],
    layouts: &[Layout],
    attributes: &HashMap<String, Attribute>,
) -> HoduResult<BackendStorage> {
    match op {
        Op::Binary(_) | Op::BinaryLogical(_) | Op::Cmp(_) | Op::CmpScalar(_) => {
            binary::execute(inputs, layouts, op, attributes)
        },

        Op::Unary(_) | Op::UnaryLogical(_) | Op::UnaryScalar(_) => unary::execute(inputs, layouts, op, attributes),

        Op::Matrix(_) => matrix::execute(inputs, layouts, op),

        Op::Reduce(_) | Op::Windowing(_) => reduce::execute(inputs, layouts, op, attributes),

        Op::Concat(_) | Op::Split(_) => concat_split::execute(inputs, layouts, op, attributes),

        Op::Indexing(_) => indexing::execute(inputs, layouts, op, attributes),

        Op::Conv(_) => conv::execute(inputs, layouts, op, attributes),

        Op::Shape(_) | Op::ShapeScalars(_) | Op::Cast(_) | Op::Memory(_) | Op::Dummy => {
            shape_cast_memory::execute(inputs, layouts, op, attributes)
        },
    }
}
