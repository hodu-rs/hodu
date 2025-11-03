mod compiler;
mod helpers;

use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    executor::{CompileOptions, ExecutionInputs, ExecutionOutputs, ExecutorT},
    script::Script,
    tensor::TensorId,
    types::{device::Device, dtype::DType, layout::Layout},
};
use helpers::xla_error_to_hodu_error;
use hodu_xla::{ElementType, PjRtClient, PjRtLoadedExecutable, XlaBuilder};
use std::collections::HashMap;
use std::sync::Arc;

// Thread-safe wrapper for PjRtLoadedExecutable
// SAFETY: XLA's PjRtLoadedExecutable is thread-safe in practice,
// but the Rust bindings don't expose Send/Sync markers
pub(super) struct ThreadSafeExecutable(PjRtLoadedExecutable);

unsafe impl Send for ThreadSafeExecutable {}
unsafe impl Sync for ThreadSafeExecutable {}

impl std::ops::Deref for ThreadSafeExecutable {
    type Target = PjRtLoadedExecutable;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Thread-safe wrapper for PjRtClient
// SAFETY: XLA's PjRtClient is thread-safe in practice,
// but the Rust bindings don't expose Send/Sync markers
struct ThreadSafeClient(PjRtClient);

unsafe impl Send for ThreadSafeClient {}
unsafe impl Sync for ThreadSafeClient {}

impl std::ops::Deref for ThreadSafeClient {
    type Target = PjRtClient;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct XlaCompiledScript {
    pub(super) executable: Arc<ThreadSafeExecutable>,
    pub(super) input_mapping: HashMap<String, TensorId>,
    pub(super) output_mapping: HashMap<String, TensorId>,
    pub(super) tensor_layouts: HashMap<TensorId, Layout>,
    pub(super) tensor_dtypes: HashMap<TensorId, DType>,
}

pub struct XlaExecutor {
    client: Arc<ThreadSafeClient>,
    device: Device,
}

impl XlaExecutor {}

unsafe impl Send for XlaExecutor {}
unsafe impl Sync for XlaExecutor {}

impl ExecutorT for XlaExecutor {
    type CompiledScript = XlaCompiledScript;

    fn backend_name(&self) -> &'static str {
        "xla"
    }

    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::CPU]
    }

    fn current_device(&self) -> Device {
        self.device
    }

    fn compile(&mut self, script: &Script, _options: CompileOptions) -> HoduResult<Self::CompiledScript> {
        let script_ir = script
            .get_ir()
            .ok_or_else(|| HoduError::ScriptValidationFailed("Script has no IR".to_string()))?;
        script_ir.validate().map_err(HoduError::ScriptValidationFailed)?;

        // Create input and output mappings exactly like HoduExecutor
        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();

        for input in &script_ir.graph.metadata.inputs {
            input_mapping.insert(input.name.clone(), input.tensor_id);
        }
        for output in &script_ir.graph.metadata.outputs {
            output_mapping.insert(output.name.clone(), output.tensor_id);
        }

        let tensor_layouts = self.collect_tensor_layouts(script_ir);
        let tensor_dtypes = self.collect_tensor_dtypes(script_ir, script);

        // Create dummy executable first
        let dummy_client = ThreadSafeClient(PjRtClient::cpu().map_err(xla_error_to_hodu_error)?);
        let dummy_builder = XlaBuilder::new("dummy");
        let dummy_param = dummy_builder
            .parameter(0, ElementType::F32, &[1], "dummy")
            .map_err(xla_error_to_hodu_error)?;
        let dummy_computation = dummy_param.build().map_err(xla_error_to_hodu_error)?;
        let dummy_executable = dummy_client
            .compile(&dummy_computation)
            .map_err(xla_error_to_hodu_error)?;

        let compiled_script = XlaCompiledScript {
            executable: Arc::new(ThreadSafeExecutable(dummy_executable)), // Dummy executable, will be replaced
            input_mapping,
            output_mapping,
            tensor_layouts,
            tensor_dtypes,
        };

        // Build the actual XLA computation
        let executable = self.build_xla_computation(&compiled_script, script_ir)?;

        Ok(XlaCompiledScript {
            executable,
            input_mapping: compiled_script.input_mapping,
            output_mapping: compiled_script.output_mapping,
            tensor_layouts: compiled_script.tensor_layouts,
            tensor_dtypes: compiled_script.tensor_dtypes,
        })
    }

    fn execute(&self, compiled: &Self::CompiledScript, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
        // Convert inputs to XLA literals using input_mapping (exactly like HoduExecutor)
        // Get input names in consistent order (sort for consistent ordering)
        let mut input_names: Vec<_> = compiled.input_mapping.keys().cloned().collect();
        input_names.sort();

        // XLA Literal is not Send, so we cannot use rayon here
        let mut xla_inputs = Vec::new();
        for input_name in input_names.iter() {
            let tensor = inputs
                .get(input_name.as_str())
                .ok_or_else(|| HoduError::InternalError(format!("Input {} not found", input_name)))?;

            let literal = self.tensor_to_literal(tensor).map_err(|e| {
                HoduError::InternalError(format!("Failed to convert input {} to literal: {}", input_name, e))
            })?;
            xla_inputs.push(literal);
        }

        // Execute the computation
        let result_buffers = compiled
            .executable
            .execute::<hodu_xla::Literal>(&xla_inputs)
            .map_err(|e| HoduError::InternalError(format!("Failed to execute XLA computation: {:?}", e)))?;

        // Convert results back to tensors using output_mapping
        let mut outputs = HashMap::new();

        // Get output names in consistent order (to match compile order)
        let mut output_names: Vec<_> = compiled.output_mapping.keys().cloned().collect();
        output_names.sort(); // Must match the order used in compile

        if output_names.len() == 1 {
            // Single output case
            let result_literal = result_buffers[0][0]
                .to_literal_sync()
                .map_err(|e| HoduError::InternalError(format!("Failed to convert result to literal: {:?}", e)))?;

            let output_name = &output_names[0];

            // Get expected dtype from tensor_dtypes mapping
            let output_tensor_id = compiled
                .output_mapping
                .get(output_name)
                .ok_or_else(|| HoduError::InternalError(format!("Output tensor ID not found for {}", output_name)))?;
            let expected_dtype = compiled
                .tensor_dtypes
                .get(output_tensor_id)
                .copied()
                .unwrap_or(DType::F32);

            // For operations that change dtype (like argmax/argmin), detect actual dtype from literal
            let actual_dtype =
                self.xla_element_type_to_dtype(result_literal.element_type().map_err(|e| {
                    HoduError::InternalError(format!("Failed to get element type from literal: {:?}", e))
                })?);
            let dtype_to_use = actual_dtype.unwrap_or(expected_dtype);

            let tensor = self
                .literal_to_tensor(&result_literal, dtype_to_use)
                .map_err(|e| HoduError::InternalError(format!("Failed to convert literal to tensor: {}", e)))?;

            outputs.insert(output_name.clone(), tensor);
        } else {
            // Multiple output case - access tuple elements directly
            if result_buffers[0].len() != output_names.len() {
                return Err(HoduError::InternalError(format!(
                    "Tuple has {} elements but expected {} outputs",
                    result_buffers[0].len(),
                    output_names.len()
                )));
            }

            // XLA result_buffers is not Send, so we cannot use rayon here
            for (i, output_name) in output_names.iter().enumerate() {
                let element_literal = result_buffers[0][i].to_literal_sync().map_err(|e| {
                    HoduError::InternalError(format!("Failed to convert tuple element {} to literal: {:?}", i, e))
                })?;

                let output_tensor_id = compiled.output_mapping.get(output_name).ok_or_else(|| {
                    HoduError::InternalError(format!("Output tensor ID not found for {}", output_name))
                })?;
                let expected_dtype = compiled
                    .tensor_dtypes
                    .get(output_tensor_id)
                    .copied()
                    .unwrap_or(DType::F32);

                let tensor = self.literal_to_tensor(&element_literal, expected_dtype).map_err(|e| {
                    HoduError::InternalError(format!("Failed to convert literal {} to tensor: {}", i, e))
                })?;

                outputs.insert(output_name.clone(), tensor);
            }
        }

        Ok(outputs)
    }

    fn cleanup(&mut self) -> HoduResult<()> {
        Ok(())
    }
}
