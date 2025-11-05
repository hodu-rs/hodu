use super::{super::context::BuilderState, super::ir::*};
use crate::{error::HoduResult, layer::compat::*, tensor::TensorId};

use super::tracing::trace_tensor_operations;

/// Build basic block from operations
pub fn build_basic_block(state: &mut BuilderState, block_id: BlockId) -> HoduResult<BasicBlock> {
    let mut block = BasicBlock::new(block_id).with_label("entry".to_string());

    // For now, we'll need to trace operations from the computation graph
    // This requires walking the tensor dependency graph
    // We'll implement a simple version that processes operations in order

    // TODO: Implement actual operation tracing from tensors
    // For now, create a placeholder that just returns outputs

    // Collect all constants (tensors with storage that aren't inputs)
    let graph_inputs = state.graph_inputs.clone();
    let graph_outputs = state.graph_outputs.clone();

    let input_ids: HashSet<TensorId> = graph_inputs.iter().map(|(_, t)| t.id()).collect();

    // Walk output tensors and trace back to inputs
    for (_, output_tensor) in &graph_outputs {
        trace_tensor_operations(state, output_tensor.id(), &mut block, &input_ids)?;
    }

    // Set terminator to return outputs
    let output_values: Vec<ValueId> = graph_outputs
        .iter()
        .filter_map(|(_, tensor)| state.tensor_to_value.get(&tensor.id()).copied())
        .collect();

    block.set_terminator(Terminator::Return { values: output_values });

    Ok(block)
}
