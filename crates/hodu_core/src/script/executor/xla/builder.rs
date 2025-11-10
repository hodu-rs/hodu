use crate::{
    error::{HoduError, HoduResult},
    script::compiler::CompiledModule,
    types::Device,
};
use hodu_xla::{PjRtClient, PjRtLoadedExecutable, XlaBuilder};
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::{Arc, LazyLock, RwLock},
};

use super::super::types::*;
use super::{
    conversion::{create_constant_op, dtype_to_element_type, literal_to_tensor, tensor_to_literal},
    ops::execute_xla_op,
};

// Thread-safe wrapper for PjRtLoadedExecutable
// SAFETY: PjRtLoadedExecutable is internally thread-safe but doesn't implement Send
struct SendExecutable(PjRtLoadedExecutable);
unsafe impl Send for SendExecutable {}
unsafe impl Sync for SendExecutable {}

// XLA executable cache for reusing compiled computations
static EXECUTABLE_CACHE: LazyLock<RwLock<HashMap<u64, Arc<SendExecutable>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Compute hash of compiled module for caching
fn compute_module_hash(compiled: &CompiledModule, device: Device) -> u64 {
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();

    // Hash device
    std::mem::discriminant(&device).hash(&mut hasher);

    // Hash execution plan (operation types and attributes)
    for instr in &compiled.execution_plan {
        std::mem::discriminant(&instr.op).hash(&mut hasher);
        instr.result.0.hash(&mut hasher);
        for input in &instr.inputs {
            input.0.hash(&mut hasher);
        }
    }

    // Hash input/output mappings
    let mut input_keys: Vec<_> = compiled.input_mapping.keys().collect();
    input_keys.sort();
    for key in input_keys {
        key.hash(&mut hasher);
        compiled.input_mapping[key].0.hash(&mut hasher);
    }

    let mut output_keys: Vec<_> = compiled.output_mapping.keys().collect();
    output_keys.sort();
    for key in output_keys {
        key.hash(&mut hasher);
        compiled.output_mapping[key].0.hash(&mut hasher);
    }

    // Hash constant data (shapes and dtypes, not actual data for performance)
    let mut constant_keys: Vec<_> = compiled.constant_data.keys().collect();
    constant_keys.sort();
    for key in constant_keys {
        let constant = &compiled.constant_data[key];
        constant.shape.size().hash(&mut hasher);
        std::mem::discriminant(&constant.dtype).hash(&mut hasher);
    }

    hasher.finish()
}

/// Build and execute XLA computation from the compiled module
/// Uses compilation cache to avoid recompiling the same computation
pub fn build_and_execute_xla(
    device: Device,
    compiled: &CompiledModule,
    inputs: &ExecutionInputs<'_>,
) -> HoduResult<ExecutionOutputs> {
    // Compute hash for cache lookup
    let cache_key = compute_module_hash(compiled, device);

    // Check if executable is cached
    let executable = {
        let cache = EXECUTABLE_CACHE
            .read()
            .map_err(|e| HoduError::InternalError(format!("Failed to read executable cache: {}", e)))?;

        if let Some(cached_exec) = cache.get(&cache_key) {
            // Cache hit - reuse compiled executable
            cached_exec.clone()
        } else {
            // Cache miss - need to compile
            drop(cache); // Release read lock before compiling

            // Create XLA client
            let client = match device {
                Device::CPU => PjRtClient::cpu()?,
                #[cfg(feature = "cuda")]
                Device::CUDA(_) => PjRtClient::gpu(0.95, true)?,
                #[cfg(feature = "metal")]
                Device::Metal => {
                    return Err(HoduError::InternalError(
                        "Metal device is not supported for XLA".to_string(),
                    ))
                },
            };

            // Build and compile computation
            let computation = build_xla_computation(compiled)?;
            let executable = Arc::new(SendExecutable(client.compile(&computation)?));

            // Store in cache
            let mut cache = EXECUTABLE_CACHE
                .write()
                .map_err(|e| HoduError::InternalError(format!("Failed to write executable cache: {}", e)))?;
            cache.insert(cache_key, executable.clone());

            executable
        }
    };

    // Convert inputs to literals
    let mut input_names: Vec<_> = compiled.input_mapping.keys().cloned().collect();
    input_names.sort();

    let input_literals: Vec<_> = input_names
        .iter()
        .filter_map(|name| {
            inputs
                .get(name.as_str())
                .and_then(|tensor| tensor_to_literal(tensor).ok())
        })
        .collect();

    // Execute cached/compiled computation
    let result_buffers = executable.0.execute::<hodu_xla::Literal>(&input_literals)?;

    // Convert results back to tensors
    let mut outputs = HashMap::new();

    let mut output_names: Vec<_> = compiled.output_mapping.keys().cloned().collect();
    output_names.sort();

    if output_names.len() == 1 {
        let result_literal = result_buffers[0][0]
            .to_literal_sync()
            .map_err(|e| HoduError::InternalError(format!("Failed to get result literal: {:?}", e)))?;

        let output_name = &output_names[0];
        let value_id = compiled
            .output_mapping
            .get(output_name)
            .ok_or_else(|| HoduError::ExecutionError(format!("missing output mapping: {}", output_name)))?;
        let dtype = compiled
            .value_dtypes
            .get(value_id)
            .copied()
            .unwrap_or(crate::types::DType::F32);

        let tensor = literal_to_tensor(&result_literal, dtype)?;
        outputs.insert(output_name.clone(), tensor);
    } else {
        for (i, output_name) in output_names.iter().enumerate() {
            let element_literal = result_buffers[0][i]
                .to_literal_sync()
                .map_err(|e| HoduError::InternalError(format!("Failed to get tuple element: {:?}", e)))?;

            let value_id = compiled
                .output_mapping
                .get(output_name)
                .ok_or_else(|| HoduError::ExecutionError(format!("missing output mapping: {}", output_name)))?;
            let dtype = compiled
                .value_dtypes
                .get(value_id)
                .copied()
                .unwrap_or(crate::types::DType::F32);

            let tensor = literal_to_tensor(&element_literal, dtype)?;
            outputs.insert(output_name.clone(), tensor);
        }
    }

    Ok(outputs)
}

/// Build XLA computation graph from compiled module
/// This is separated from execution to enable caching
fn build_xla_computation(compiled: &CompiledModule) -> HoduResult<hodu_xla::XlaComputation> {
    let builder = XlaBuilder::new("computation");
    let mut xla_ops = HashMap::new();

    // Create parameters for inputs
    let mut input_names: Vec<_> = compiled.input_mapping.keys().cloned().collect();
    input_names.sort();

    for (i, input_name) in input_names.iter().enumerate() {
        if let Some(&value_id) = compiled.input_mapping.get(input_name) {
            // Get layout and dtype
            let layout = compiled
                .value_layouts
                .get(&value_id)
                .ok_or_else(|| HoduError::ExecutionError(format!("missing layout for input: {}", input_name)))?;
            let dtype = compiled
                .value_dtypes
                .get(&value_id)
                .copied()
                .unwrap_or(crate::types::DType::F32);

            // Convert dtype to ElementType
            let element_type = dtype_to_element_type(dtype)?;
            let dims: Vec<i64> = layout.shape().dims().iter().map(|&d| d as i64).collect();

            let param = builder.parameter(i as i64, element_type, &dims, &format!("input_{}", i))?;

            xla_ops.insert(value_id, param);
        }
    }

    // Process constants
    for (tensor_id, constant_data) in &compiled.constant_data {
        // Find value_id for this tensor_id
        if let Some((&value_id, _)) = compiled.value_to_tensor.iter().find(|(_, &tid)| tid == *tensor_id) {
            // Create constant XLA op
            let constant_op = create_constant_op(&builder, constant_data)?;
            xla_ops.insert(value_id, constant_op);
        }
    }

    // Execute instructions
    for instr in &compiled.execution_plan {
        // Skip constant loads
        if let Some(crate::script::builder::ir::Attribute::Bool(true)) = instr.attributes.get("is_constant") {
            continue;
        }

        // Get input ops
        let input_ops: Vec<_> = instr
            .inputs
            .iter()
            .filter_map(|vid| xla_ops.get(vid).cloned())
            .collect();

        if input_ops.len() != instr.inputs.len() {
            return Err(HoduError::InternalError(format!(
                "Missing input ops for instruction, expected {}, got {}",
                instr.inputs.len(),
                input_ops.len()
            )));
        }

        // Execute operation
        let result_op = execute_xla_op(&builder, &instr.op, &input_ops, &instr.attributes, compiled, instr)?;
        xla_ops.insert(instr.result, result_op);
    }

    // Build output computation
    let mut output_names: Vec<_> = compiled.output_mapping.keys().cloned().collect();
    output_names.sort();

    let computation = if output_names.len() == 1 {
        // Single output
        let output_name = &output_names[0];
        let value_id = compiled
            .output_mapping
            .get(output_name)
            .ok_or_else(|| HoduError::ExecutionError(format!("missing output mapping: {}", output_name)))?;
        let output_op = xla_ops
            .get(value_id)
            .ok_or_else(|| HoduError::ExecutionError(format!("missing output op for: {}", output_name)))?;
        output_op.build()?
    } else {
        // Multiple outputs - create tuple
        let output_ops: Vec<_> = output_names
            .iter()
            .filter_map(|name| {
                compiled
                    .output_mapping
                    .get(name)
                    .and_then(|vid| xla_ops.get(vid).cloned())
            })
            .collect();

        if output_ops.len() != output_names.len() {
            return Err(HoduError::InternalError("Missing some output ops".to_string()));
        }

        let tuple_op = builder
            .tuple(&output_ops)
            .map_err(|e| HoduError::InternalError(format!("Failed to create tuple: {:?}", e)))?;
        tuple_op.build()?
    };

    Ok(computation)
}
