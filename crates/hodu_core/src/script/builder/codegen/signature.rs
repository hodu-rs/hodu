use super::{super::context::BuilderState, super::ir::*};
use crate::{compat::*, error::HoduResult};

use super::tracing::allocate_value_id;

/// Build function signature from inputs/outputs
pub fn build_signature(state: &mut BuilderState) -> HoduResult<FunctionSignature> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    // Clone to avoid borrow issues
    let graph_inputs = state.graph_inputs.clone();
    let graph_outputs = state.graph_outputs.clone();

    // Process inputs
    for (name, tensor) in &graph_inputs {
        let value_id = allocate_value_id(state, tensor.id());
        let layout = tensor.layout();

        let param = Parameter::new(name.to_string(), value_id)
            .with_shape(layout.shape().clone())
            .with_dtype(tensor.dtype());

        inputs.push(param);
    }

    // Process outputs
    for (name, tensor) in &graph_outputs {
        let value_id = allocate_value_id(state, tensor.id());
        let layout = tensor.layout();

        let param = Parameter::new(name.to_string(), value_id)
            .with_shape(layout.shape().clone())
            .with_dtype(tensor.dtype());

        outputs.push(param);
    }

    Ok(FunctionSignature::new(inputs, outputs))
}
