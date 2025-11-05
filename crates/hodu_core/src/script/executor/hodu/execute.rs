use super::{
    ops,
    types::{ExecutionInputs, ExecutionOutputs},
};
use crate::{
    be::storage::BackendStorage,
    error::{HoduError, HoduResult},
    layer::compat::*,
    script::{
        builder::ir::{Attribute, ValueId},
        compiler::CompiledModule,
    },
    tensor::from_storage,
    types::{Device, Layout},
};

/// Execute a compiled module with the given inputs
pub fn execute(device: Device, compiled: &CompiledModule, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
    validate_inputs(compiled, &inputs)?;

    // Map value_id to storage for runtime
    let mut value_storage: HashMap<ValueId, Arc<BackendStorage>> = HashMap::new();

    // Load input values
    for (name, value_id) in &compiled.input_mapping {
        let tensor = inputs
            .get(name.as_str())
            .ok_or_else(|| HoduError::InternalError(format!("Missing input: {}", name)))?;

        let storage = tensor.with_storage(|s| Ok(Arc::new(s.clone())))?;

        value_storage.insert(*value_id, storage);
    }

    // Execute instructions in order
    for instr in &compiled.execution_plan {
        // Check if this is a constant load
        if let Some(Attribute::Bool(true)) = instr.attributes.get("is_constant") {
            // Load constant from value_to_tensor mapping
            if let Some(tensor_id) = compiled.value_to_tensor.get(&instr.result) {
                // Get the tensor and extract storage
                let tensor = crate::tensor::tensor_from_id(*tensor_id);
                let storage = if tensor.device() != device {
                    // Convert to target device
                    let cpu_storage = tensor.with_storage(|s| s.to_cpu_storage())?;
                    let new_storage = match device {
                        Device::CPU => BackendStorage::CPU(cpu_storage),
                        #[cfg(feature = "metal")]
                        Device::Metal => BackendStorage::Metal(
                            crate::be_metal::storage::MetalStorage::from_cpu_storage(&cpu_storage)?,
                        ),
                        #[allow(unreachable_patterns)]
                        _ => return Err(HoduError::InternalError(format!("Unsupported device: {:?}", device))),
                    };
                    Arc::new(new_storage)
                } else {
                    tensor.with_storage(|s| Ok(Arc::new(s.clone())))?
                };
                value_storage.insert(instr.result, storage);
                continue;
            }
            return Err(HoduError::InternalError("Failed to load constant".to_string()));
        }

        // Get input storages for this operation
        let input_storages: Vec<Arc<BackendStorage>> = instr
            .inputs
            .iter()
            .filter_map(|vid| value_storage.get(vid).cloned())
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
        let input_layouts: Vec<Layout> = instr
            .inputs
            .iter()
            .filter_map(|vid| compiled.value_layouts.get(vid).cloned())
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
        value_storage.insert(instr.result, Arc::new(result_storage));
    }

    // Collect outputs
    let mut outputs = HashMap::new();
    for (name, value_id) in &compiled.output_mapping {
        let storage = value_storage
            .get(value_id)
            .ok_or_else(|| HoduError::InternalError(format!("Missing output: {}", name)))?;

        let layout = compiled
            .value_layouts
            .get(value_id)
            .ok_or_else(|| HoduError::InternalError(format!("Missing layout for output: {}", name)))?
            .clone();

        let tensor = from_storage((**storage).clone(), layout, true, false);
        outputs.insert(name.clone(), tensor);
    }

    Ok(outputs)
}

/// Validate that inputs match the compiled module's requirements
pub fn validate_inputs(compiled: &CompiledModule, inputs: &ExecutionInputs<'_>) -> HoduResult<()> {
    for name in compiled.input_mapping.keys() {
        if !inputs.contains_key(name.as_str()) {
            return Err(HoduError::InternalError(format!("Missing required input: {}", name)));
        }
    }
    Ok(())
}
