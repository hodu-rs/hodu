use super::{
    ops,
    types::{ExecutionInputs, ExecutionOutputs},
};
use crate::{
    be::storage::BackendStorage,
    compat::*,
    error::{HoduError, HoduResult},
    script::{builder::ir::Attribute, compiler::CompiledModule},
    tensor::from_storage,
    types::Layout,
};

/// Execute a compiled module with the given inputs
pub fn execute(compiled: &CompiledModule, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
    validate_inputs(compiled, &inputs)?;

    // Use Vec instead of HashMap for O(1) access
    // max_value_id is pre-computed during compilation for efficiency
    let mut value_storage: Vec<Option<Arc<BackendStorage>>> = vec![None; compiled.max_value_id + 1];

    // Load input values
    for (name, value_id) in &compiled.input_mapping {
        let tensor = inputs
            .get(name.as_str())
            .ok_or_else(|| HoduError::MissingInput(name.clone()))?;

        let storage = tensor.with_storage(|s| Ok(Arc::new(s.clone())))?;

        value_storage[value_id.0] = Some(storage);
    }

    // Execute instructions in order
    for instr in &compiled.execution_plan {
        // Check if this is a constant load
        if let Some(Attribute::Bool(true)) = instr.attributes.get("is_constant") {
            // Load constant from pre-converted storage cache
            if let Some(tensor_id) = compiled.value_to_tensor.get(&instr.result) {
                if let Some(storage) = compiled.constant_storages.get(tensor_id) {
                    // Use pre-converted storage (no conversion needed!)
                    value_storage[instr.result.0] = Some(storage.clone());
                    continue;
                }
            }
            return Err(HoduError::InternalError(
                "Failed to load constant from cache".to_string(),
            ));
        }

        // Get input storages for this operation
        let input_storages: Vec<&Arc<BackendStorage>> = instr
            .inputs
            .iter()
            .filter_map(|vid| value_storage[vid.0].as_ref())
            .collect();

        if input_storages.len() != instr.inputs.len() {
            return Err(HoduError::InternalError(format!(
                "Missing inputs for operation: {:?}. Expected {}, got {}",
                instr.op,
                instr.inputs.len(),
                input_storages.len()
            )));
        }

        // Get input layouts
        let input_layouts: Vec<&Layout> = instr
            .inputs
            .iter()
            .filter_map(|vid| compiled.value_layouts.get(vid))
            .collect();

        if input_layouts.len() != instr.inputs.len() {
            return Err(HoduError::InternalError(format!(
                "Missing layouts for operation: {:?}. Expected {}, got {}",
                instr.op,
                instr.inputs.len(),
                input_layouts.len()
            )));
        }

        // Execute the operation
        let result_storage = ops::execute_operation(&instr.op, &input_storages, &input_layouts, &instr.attributes)?;
        value_storage[instr.result.0] = Some(Arc::new(result_storage));
    }

    // Collect outputs
    let mut outputs = HashMap::new();
    for (name, value_id) in &compiled.output_mapping {
        let storage = value_storage[value_id.0]
            .as_ref()
            .ok_or_else(|| HoduError::ExecutionError(format!("missing output: {}", name)))?;

        let layout = compiled
            .value_layouts
            .get(value_id)
            .ok_or_else(|| HoduError::ExecutionError(format!("missing layout for output: {}", name)))?
            .clone();

        let tensor = from_storage((**storage).clone(), layout, true, false, None);
        outputs.insert(name.clone(), tensor);
    }

    Ok(outputs)
}

/// Validate that inputs match the compiled module's requirements
pub fn validate_inputs(compiled: &CompiledModule, inputs: &ExecutionInputs<'_>) -> HoduResult<()> {
    for name in compiled.input_mapping.keys() {
        if !inputs.contains_key(name.as_str()) {
            return Err(HoduError::MissingInput(name.clone()));
        }
    }
    Ok(())
}
