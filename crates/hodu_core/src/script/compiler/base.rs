use crate::{
    compat::*,
    error::HoduResult,
    script::builder::ir::*,
    tensor::TensorId,
    types::{DType, Layout},
};

use super::types::*;

/// Finds the main function in a module.
/// Returns the first function named "forward" or ending with "_main".
pub fn find_main_function(module: &Module) -> Option<&Function> {
    module
        .functions
        .iter()
        .find(|f| f.name == "forward" || f.name.ends_with("_main"))
}

/// Builds the execution plan from a function.
/// This traverses all blocks and instructions, converting them into compiled instructions.
pub fn build_execution_plan(
    function: &Function,
    value_to_tensor: &mut HashMap<ValueId, TensorId>,
) -> HoduResult<Vec<CompiledInstruction>> {
    let mut execution_plan = Vec::new();

    // Traverse blocks in order
    for block in &function.blocks {
        for instruction in &block.instructions {
            match instruction {
                Instruction::Compute {
                    result,
                    op,
                    inputs,
                    attributes,
                } => {
                    execution_plan.push(CompiledInstruction {
                        op: op.clone(),
                        inputs: inputs.clone(),
                        result: *result,
                        attributes: attributes.clone(),
                    });
                },
                Instruction::LoadConstant { result, tensor_id } => {
                    // Store the mapping from value_id to tensor_id
                    value_to_tensor.insert(*result, *tensor_id);

                    // Mark as constant load operation
                    let mut attrs = HashMap::new();
                    attrs.insert("is_constant".to_string(), Attribute::Bool(true));

                    execution_plan.push(CompiledInstruction {
                        op: crate::ops::Op::Dummy,
                        inputs: vec![],
                        result: *result,
                        attributes: attrs,
                    });
                },
                Instruction::Phi { .. } => {
                    // TODO: Handle Phi nodes properly
                },
            }
        }
    }

    Ok(execution_plan)
}

/// Extracts input and output mappings from function signature.
/// Returns (input_mapping, output_mapping) where keys are parameter names
/// and values are their corresponding ValueIds.
pub fn extract_input_output_mapping(function: &Function) -> (HashMap<String, ValueId>, HashMap<String, ValueId>) {
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for param in &function.signature.inputs {
        input_mapping.insert(param.name.clone(), param.value_id);
    }

    for param in &function.signature.outputs {
        output_mapping.insert(param.name.clone(), param.value_id);
    }

    (input_mapping, output_mapping)
}

/// Extracts layout and dtype information for all values in the function.
/// Returns (value_layouts, value_dtypes) containing the collected metadata.
pub fn extract_value_info(function: &Function) -> (HashMap<ValueId, Layout>, HashMap<ValueId, DType>) {
    let mut value_layouts = HashMap::new();
    let mut value_dtypes = HashMap::new();

    // From function signature
    for param in &function.signature.inputs {
        if let Some(ref shape) = param.shape {
            value_layouts.insert(param.value_id, Layout::from_shape(shape));
        }
        if let Some(dtype) = param.dtype {
            value_dtypes.insert(param.value_id, dtype);
        }
    }

    for param in &function.signature.outputs {
        if let Some(ref shape) = param.shape {
            value_layouts.insert(param.value_id, Layout::from_shape(shape));
        }
        if let Some(dtype) = param.dtype {
            value_dtypes.insert(param.value_id, dtype);
        }
    }

    // From value info
    for (value_id, info) in &function.value_info {
        if let Some(ref layout) = info.layout {
            value_layouts.insert(*value_id, layout.clone());
        }
        if let Some(dtype) = info.dtype {
            value_dtypes.insert(*value_id, dtype);
        }
    }

    (value_layouts, value_dtypes)
}
