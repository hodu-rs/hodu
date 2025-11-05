use super::{super::context::BuilderState, super::ir::*};
use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    tensor::TensorId,
    types::Layout,
};

use super::constants::load_constant;

/// Trace tensor operations recursively
pub fn trace_tensor_operations(
    state: &mut BuilderState,
    tensor_id: TensorId,
    block: &mut BasicBlock,
    input_ids: &HashSet<TensorId>,
) -> HoduResult<ValueId> {
    // If already traced, return existing value
    if let Some(&value_id) = state.tensor_to_value.get(&tensor_id) {
        return Ok(value_id);
    }

    // If it's an input, just allocate value id
    if input_ids.contains(&tensor_id) {
        let value_id = allocate_value_id(state, tensor_id);
        return Ok(value_id);
    }

    // Check if tensor has storage (constant)
    if let Some(_tensor_ref) = crate::tensor::get(tensor_id) {
        let tensor = crate::tensor::tensor_from_id(tensor_id);
        if tensor.has_storage() && !tensor.is_runtime() {
            // This is a constant tensor (weight/bias)
            return load_constant(state, block, tensor_id);
        }
    }

    // TODO: Trace computation graph
    // For now, we need access to the operation that produced this tensor
    // This requires storing operation info in the tensor or maintaining a separate graph

    // Placeholder: just allocate a value
    let value_id = allocate_value_id(state, tensor_id);
    Ok(value_id)
}

/// Generate SSA instruction for an operation
#[allow(dead_code)]
pub fn emit_operation(
    state: &mut BuilderState,
    block: &mut BasicBlock,
    op: Op,
    input_tensors: &[TensorId],
    output_tensors: &[TensorId],
    _input_layouts: &[Layout],
    _output_layouts: &[Layout],
) -> HoduResult<Vec<ValueId>> {
    // Trace input tensors first
    let mut input_values = Vec::new();
    for &input_id in input_tensors {
        let value_id = state
            .tensor_to_value
            .get(&input_id)
            .copied()
            .ok_or_else(|| HoduError::InternalError(format!("Input tensor {:?} not found", input_id)))?;
        input_values.push(value_id);
    }

    // Allocate output values
    let mut output_values = Vec::new();
    for &output_id in output_tensors {
        let value_id = allocate_value_id(state, output_id);
        output_values.push(value_id);
    }

    // For now, assume single output
    if !output_values.is_empty() {
        let result = output_values[0];

        // Convert operation-specific attributes to generic attributes
        let attributes = HashMap::new();
        // TODO: Extract attributes from Op variants

        block.add_instruction(Instruction::Compute {
            result,
            op,
            inputs: input_values,
            attributes,
        });
    }

    Ok(output_values)
}

/// Allocate a new ValueId for a TensorId
pub fn allocate_value_id(state: &mut BuilderState, tensor_id: TensorId) -> ValueId {
    if let Some(&existing) = state.tensor_to_value.get(&tensor_id) {
        return existing;
    }

    let value_id = ValueId(state.value_counter);
    state.value_counter += 1;
    state.tensor_to_value.insert(tensor_id, value_id);
    value_id
}
