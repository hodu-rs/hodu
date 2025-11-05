use crate::{
    be::storage::BackendStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{MatrixOp, Op},
    types::Layout,
};

/// Execute matrix operations: Matmul, Dot
pub fn execute(inputs: &[Arc<BackendStorage>], layouts: &[Layout], op: &Op) -> HoduResult<BackendStorage> {
    match op {
        Op::Matrix(MatrixOp::Matmul) => {
            if inputs.len() != 2 || layouts.len() != 2 {
                return Err(HoduError::InternalError(format!(
                    "Matmul operation requires 2 inputs and layouts, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            inputs[0].call_matmul(&inputs[1], &layouts[0], &layouts[1], op.clone())
        },

        Op::Matrix(MatrixOp::Dot) => {
            if inputs.len() != 2 || layouts.len() != 2 {
                return Err(HoduError::InternalError(format!(
                    "Dot operation requires 2 inputs and layouts, got {} and {}",
                    inputs.len(),
                    layouts.len()
                )));
            }
            inputs[0].call_dot(&inputs[1], &layouts[0], &layouts[1], op.clone())
        },

        _ => Err(HoduError::InternalError(format!(
            "Unsupported matrix operation: {:?}",
            op
        ))),
    }
}
