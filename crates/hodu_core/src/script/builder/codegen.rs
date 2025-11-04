use super::{context::BuilderState, ir::*};
use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    tensor::TensorId,
    types::Layout,
};

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
                        if !state.tensor_to_value.contains_key(&tensor.id()) {
                            if let Some(_) = crate::tensor::get(tensor.id()) {
                                if tensor.has_storage() && !tensor.is_runtime() {
                                    return Some(tensor.id());
                                }
                            }
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

/// Build function signature from inputs/outputs
fn build_signature(state: &mut BuilderState) -> HoduResult<FunctionSignature> {
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

/// Build basic block from operations
fn build_basic_block(state: &mut BuilderState, block_id: BlockId) -> HoduResult<BasicBlock> {
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

/// Trace tensor operations recursively
fn trace_tensor_operations(
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

/// Load constant tensor
fn load_constant(state: &mut BuilderState, block: &mut BasicBlock, tensor_id: TensorId) -> HoduResult<ValueId> {
    // Add constant data to module if not already present
    if !state.module.constants.contains_key(&tensor_id) {
        let tensor = crate::tensor::tensor_from_id(tensor_id);
        let layout = tensor.layout();

        // Get CPU storage data
        let cpu_storage = tensor.with_storage(|storage| storage.to_cpu_storage())?;

        let constant = ConstantData {
            tensor_id,
            shape: layout.shape().clone(),
            dtype: tensor.dtype(),
            data: cpu_storage.to_bytes(),
            compression: None,
        };

        state.module.add_constant(tensor_id, constant);
    }

    // Allocate value and emit load instruction
    let value_id = allocate_value_id(state, tensor_id);
    block.add_instruction(Instruction::LoadConstant {
        result: value_id,
        tensor_id,
    });

    Ok(value_id)
}

/// Generate SSA instruction for an operation
#[allow(dead_code)]
fn emit_operation(
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
fn allocate_value_id(state: &mut BuilderState, tensor_id: TensorId) -> ValueId {
    if let Some(&existing) = state.tensor_to_value.get(&tensor_id) {
        return existing;
    }

    let value_id = ValueId(state.value_counter);
    state.value_counter += 1;
    state.tensor_to_value.insert(tensor_id, value_id);
    value_id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_empty_module() {
        let mut state = BuilderState {
            name: "test".to_string(),
            module: Module::new("test".to_string()),
            current_function: None,
            current_block: None,
            value_counter: 0,
            block_counter: 0,
            tensor_to_value: HashMap::new(),
            graph_inputs: Vec::new(),
            graph_outputs: Vec::new(),
            is_ended: false,
        };

        // Can't build without inputs/outputs
        // This would fail, but demonstrates the structure
    }
}
