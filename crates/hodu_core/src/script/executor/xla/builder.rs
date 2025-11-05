use crate::{
    error::{HoduError, HoduResult},
    script::compiler::CompiledModule,
    types::Device,
};
use hodu_xla::{PjRtClient, XlaBuilder};
use std::collections::HashMap;

use super::super::types::*;
use super::{
    conversion::{create_constant_op, dtype_to_element_type, literal_to_tensor, tensor_to_literal},
    ops::execute_xla_op,
};

/// Build and execute XLA computation from the compiled module
/// This is called at runtime for each execution
pub fn build_and_execute_xla(
    device: Device,
    compiled: &CompiledModule,
    inputs: &ExecutionInputs<'_>,
) -> HoduResult<ExecutionOutputs> {
    // Create XLA client
    let client = match device {
        Device::CPU => PjRtClient::cpu()
            .map_err(|e| HoduError::InternalError(format!("Failed to create XLA CPU client: {:?}", e)))?,
        #[cfg(feature = "cuda")]
        Device::CUDA(_) => PjRtClient::gpu(0.95, true)
            .map_err(|e| HoduError::InternalError(format!("Failed to create XLA GPU client: {:?}", e)))?,
        #[cfg(any(feature = "cuda", feature = "metal"))]
        _ => {
            return Err(HoduError::InternalError(format!(
                "Device {:?} not supported for XLA",
                device
            )))
        },
    };

    // Build XLA computation
    let builder = XlaBuilder::new("computation");
    let mut xla_ops = HashMap::new();

    // Create parameters for inputs
    let mut input_names: Vec<_> = compiled.input_mapping.keys().cloned().collect();
    input_names.sort();

    for (i, input_name) in input_names.iter().enumerate() {
        if let Some(&value_id) = compiled.input_mapping.get(input_name) {
            let _tensor = inputs
                .get(input_name.as_str())
                .ok_or_else(|| HoduError::InternalError(format!("Missing input: {}", input_name)))?;

            // Get layout and dtype
            let layout = compiled
                .value_layouts
                .get(&value_id)
                .ok_or_else(|| HoduError::InternalError(format!("Missing layout for input: {}", input_name)))?;
            let dtype = compiled
                .value_dtypes
                .get(&value_id)
                .copied()
                .unwrap_or(crate::types::DType::F32);

            // Convert dtype to ElementType
            let element_type = dtype_to_element_type(dtype)?;
            let dims: Vec<i64> = layout.shape().dims().iter().map(|&d| d as i64).collect();

            let param = builder
                .parameter(i as i64, element_type, &dims, &format!("input_{}", i))
                .map_err(|e| HoduError::InternalError(format!("Failed to create parameter: {:?}", e)))?;

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
            .ok_or_else(|| HoduError::InternalError(format!("Missing output mapping: {}", output_name)))?;
        let output_op = xla_ops
            .get(value_id)
            .ok_or_else(|| HoduError::InternalError(format!("Missing output op for: {}", output_name)))?;
        output_op
            .build()
            .map_err(|e| HoduError::InternalError(format!("Failed to build XLA computation: {:?}", e)))?
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
        tuple_op
            .build()
            .map_err(|e| HoduError::InternalError(format!("Failed to build XLA computation: {:?}", e)))?
    };

    // Compile and execute
    let executable = client
        .compile(&computation)
        .map_err(|e| HoduError::InternalError(format!("Failed to compile XLA computation: {:?}", e)))?;

    // Convert inputs to literals
    let input_literals: Vec<_> = input_names
        .iter()
        .filter_map(|name| {
            inputs
                .get(name.as_str())
                .and_then(|tensor| tensor_to_literal(tensor).ok())
        })
        .collect();

    // Execute
    let result_buffers = executable
        .execute::<hodu_xla::Literal>(&input_literals)
        .map_err(|e| HoduError::InternalError(format!("Failed to execute XLA computation: {:?}", e)))?;

    // Convert results back to tensors
    let mut outputs = HashMap::new();

    if output_names.len() == 1 {
        let result_literal = result_buffers[0][0]
            .to_literal_sync()
            .map_err(|e| HoduError::InternalError(format!("Failed to get result literal: {:?}", e)))?;

        let output_name = &output_names[0];
        let value_id = compiled
            .output_mapping
            .get(output_name)
            .ok_or_else(|| HoduError::InternalError(format!("Missing output mapping: {}", output_name)))?;
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
                .ok_or_else(|| HoduError::InternalError(format!("Missing output mapping: {}", output_name)))?;
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
