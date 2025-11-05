use super::{context::BuilderState, ir::*};
use crate::{error::HoduResult, layer::compat::*, tensor::TensorId};

mod blocks;
mod constants;
mod signature;
mod tracing;

pub use blocks::build_basic_block;
pub use constants::load_constant;
pub use signature::build_signature;
pub use tracing::{allocate_value_id, emit_operation, trace_tensor_operations};

/// Build IR Module from BuilderState
pub fn build_module(state: &mut BuilderState) -> HoduResult<Module> {
    // Use existing module that already has operations recorded
    let mut module = state.module.clone();

    // Update function signature with proper inputs and outputs
    let signature = build_signature(state)?;

    if let Some(fn_name) = &state.current_function {
        if let Some(function) = module.get_function_mut(fn_name) {
            function.signature = signature;

            // Set return terminator for the current block
            if let Some(block_id) = state.current_block {
                // For each output, ensure it has a value_id
                // If output is a constant (has storage but not runtime), load it
                let outputs_to_load: Vec<TensorId> = state
                    .graph_outputs
                    .iter()
                    .filter_map(|(_, tensor)| {
                        if !state.tensor_to_value.contains_key(&tensor.id())
                            && crate::tensor::get(tensor.id()).is_some()
                            && tensor.has_storage()
                            && !tensor.is_runtime()
                        {
                            return Some(tensor.id());
                        }
                        None
                    })
                    .collect();

                // Load constants
                for tensor_id in outputs_to_load {
                    if let Some(block) = function.get_block_mut(block_id) {
                        load_constant(state, block, tensor_id)?;
                    }
                }

                let output_values: Vec<ValueId> = state
                    .graph_outputs
                    .iter()
                    .filter_map(|(_, tensor)| state.tensor_to_value.get(&tensor.id()).copied())
                    .collect();

                if let Some(block) = function.get_block_mut(block_id) {
                    block.set_terminator(Terminator::Return { values: output_values });
                }
            }

            // Copy constants to module
            function.value_info = state
                .tensor_to_value
                .iter()
                .filter_map(|(tensor_id, value_id)| {
                    crate::tensor::get(*tensor_id).map(|_| {
                        let tensor = crate::tensor::tensor_from_id(*tensor_id);
                        let layout = tensor.layout();
                        let info = ValueInfo::new(*value_id)
                            .with_shape(layout.shape().clone())
                            .with_dtype(tensor.dtype())
                            .with_layout(layout.clone());
                        (*value_id, info)
                    })
                })
                .collect();

            // Copy constant data from state.module
            module.constants = state.module.constants.clone();
        }
    }

    Ok(module)
}

/// Build a function from builder state
#[allow(dead_code)]
fn build_function(state: &mut BuilderState) -> HoduResult<Function> {
    // Reset counters
    state.value_counter = 0;
    state.block_counter = 0;
    state.tensor_to_value.clear();

    // Build function signature
    let signature = build_signature(state)?;

    // Create entry block
    let entry_block_id = BlockId(state.block_counter);
    state.block_counter += 1;
    state.current_block = Some(entry_block_id);

    let mut function = Function::new("forward".to_string(), signature, entry_block_id);

    // Build basic block with all operations
    let block = build_basic_block(state, entry_block_id)?;
    function.add_block(block);

    // Copy value info
    function.value_info = state
        .tensor_to_value
        .iter()
        .filter_map(|(tensor_id, value_id)| {
            // Get tensor info if available
            crate::tensor::get(*tensor_id).map(|_tensor_ref| {
                let tensor = crate::tensor::tensor_from_id(*tensor_id);
                let layout = tensor.layout();
                let info = ValueInfo::new(*value_id)
                    .with_shape(layout.shape().clone())
                    .with_dtype(tensor.dtype())
                    .with_layout(layout.clone());
                (*value_id, info)
            })
        })
        .collect();

    Ok(function)
}
