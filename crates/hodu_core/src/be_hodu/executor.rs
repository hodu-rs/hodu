mod compiler;
mod runtime;

use crate::{
    be_hodu::storage::HoduStorage,
    compat::*,
    error::{HoduError, HoduResult},
    executor::{CompileOptions, ExecutionInputs, ExecutionOutputs, ExecutorT},
    op::Op,
    script::{ir::NodeId, Script},
    tensor::{from_storage, TensorId},
    types::{device::Device, dtype::DType, layout::Layout},
};

type SharedStorage = Arc<HoduStorage>;

#[derive(Debug)]
pub struct HoduExecutor {
    current_device: Device,
}

#[derive(Debug)]
pub struct HoduCompiledScript {
    execution_plan: Vec<CompiledNode>,
    input_mapping: HashMap<String, TensorId>,
    output_mapping: HashMap<String, TensorId>,
    constant_storage: HashMap<TensorId, SharedStorage>,
    tensor_layouts: HashMap<TensorId, Layout>,
    tensor_dtypes: HashMap<TensorId, DType>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct CompiledNode {
    id: NodeId,
    operation: Op,
    input_tensors: Vec<TensorId>,
    output_tensors: Vec<TensorId>,
    input_layouts: Vec<Layout>,
    output_layouts: Vec<Layout>,
}

impl HoduExecutor {
    pub fn new(device: Device) -> Self {
        Self { current_device: device }
    }
}

impl ExecutorT for HoduExecutor {
    type CompiledScript = HoduCompiledScript;

    fn backend_name(&self) -> &'static str {
        "hodu"
    }

    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::CPU]
    }

    fn current_device(&self) -> Device {
        self.current_device
    }

    fn compile(&mut self, script: &Script, _options: CompileOptions) -> HoduResult<Self::CompiledScript> {
        let script_ir = script
            .get_ir()
            .ok_or_else(|| HoduError::ScriptValidationFailed("Script has no IR".to_string()))?;
        script_ir.validate().map_err(HoduError::ScriptValidationFailed)?;

        let execution_plan = self.convert_script_ir_to_compiled_nodes(script_ir)?;

        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();

        for input in &script_ir.graph.metadata.inputs {
            input_mapping.insert(input.name.clone(), input.tensor_id);
        }
        for output in &script_ir.graph.metadata.outputs {
            output_mapping.insert(output.name.clone(), output.tensor_id);
        }
        let constant_storage = self.prepare_constant_storage(script_ir)?;
        let tensor_layouts = self.collect_tensor_layouts(script_ir);
        let tensor_dtypes = self.collect_tensor_dtypes(script_ir, script);

        Ok(HoduCompiledScript {
            execution_plan,
            input_mapping,
            output_mapping,
            constant_storage,
            tensor_layouts,
            tensor_dtypes,
        })
    }

    fn execute(&self, compiled: &Self::CompiledScript, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
        self.validate_inputs(compiled, &inputs)?;

        // Pre-allocate HashMap with estimated capacity (std only)
        #[cfg(feature = "std")]
        let mut tensor_storage: HashMap<TensorId, SharedStorage> = {
            let estimated_capacity = compiled.constant_storage.len() + inputs.len() + compiled.execution_plan.len();
            HashMap::with_capacity(estimated_capacity)
        };
        #[cfg(not(feature = "std"))]
        let mut tensor_storage: HashMap<TensorId, SharedStorage> = HashMap::new();

        // Insert constant storage (already wrapped in Arc - no cloning needed)
        for (&tensor_id, shared_storage) in &compiled.constant_storage {
            tensor_storage.insert(tensor_id, shared_storage.clone());
        }

        // Convert input tensors to storage
        let input_storages = convert_inputs_to_storage(self, &inputs, compiled)?;
        for (tensor_id, storage) in input_storages {
            tensor_storage.insert(tensor_id, storage);
        }

        // Execute computation graph
        for compiled_node in &compiled.execution_plan {
            let result_storage = self.execute_node(compiled_node, &tensor_storage, compiled)?;

            // Insert result for all output tensors of this node
            let shared_result = Arc::new(result_storage);
            for &output_tensor_id in &compiled_node.output_tensors {
                tensor_storage.insert(output_tensor_id, shared_result.clone());
            }
        }

        // Prepare final outputs with pre-allocated capacity (std only)
        #[cfg(feature = "std")]
        let mut outputs = HashMap::with_capacity(compiled.output_mapping.len());
        #[cfg(not(feature = "std"))]
        let mut outputs = HashMap::new();

        let output_tensors = convert_storage_to_outputs(&tensor_storage, compiled)?;
        for (output_name, tensor) in output_tensors {
            outputs.insert(output_name, tensor);
        }

        Ok(outputs)
    }

    fn cleanup(&mut self) -> HoduResult<()> {
        // Nothing to cleanup for now
        Ok(())
    }
}

// Helper functions with function-level cfg

#[cfg(all(feature = "std", feature = "rayon"))]
fn convert_inputs_to_storage(
    executor: &HoduExecutor,
    inputs: &ExecutionInputs<'_>,
    compiled: &HoduCompiledScript,
) -> HoduResult<Vec<(TensorId, SharedStorage)>> {
    use rayon::prelude::*;

    inputs
        .par_iter()
        .filter_map(|(input_name, input_tensor)| {
            compiled.input_mapping.get(*input_name).map(|&tensor_id| {
                executor
                    .tensor_to_storage(input_tensor)
                    .map(|storage| (tensor_id, Arc::new(storage)))
            })
        })
        .collect::<HoduResult<Vec<_>>>()
}

#[cfg(not(all(feature = "std", feature = "rayon")))]
fn convert_inputs_to_storage(
    executor: &HoduExecutor,
    inputs: &ExecutionInputs<'_>,
    compiled: &HoduCompiledScript,
) -> HoduResult<Vec<(TensorId, SharedStorage)>> {
    let mut input_storages = Vec::new();
    for (input_name, input_tensor) in inputs {
        if let Some(&tensor_id) = compiled.input_mapping.get(*input_name) {
            let storage = executor.tensor_to_storage(input_tensor)?;
            input_storages.push((tensor_id, Arc::new(storage)));
        }
    }
    Ok(input_storages)
}

#[cfg(all(feature = "std", feature = "rayon"))]
fn convert_storage_to_outputs(
    tensor_storage: &HashMap<TensorId, SharedStorage>,
    compiled: &HoduCompiledScript,
) -> HoduResult<Vec<(String, crate::tensor::Tensor)>> {
    use rayon::prelude::*;

    compiled
        .output_mapping
        .par_iter()
        .map(|(output_name, &tensor_id)| {
            let storage = tensor_storage.get(&tensor_id).ok_or_else(|| {
                HoduError::InternalError(format!("Storage not found for output tensor {tensor_id:?}"))
            })?;
            let layout = compiled
                .tensor_layouts
                .get(&tensor_id)
                .ok_or_else(|| HoduError::InternalError(format!("Layout not found for output tensor {tensor_id:?}")))?;

            let output_storage = match Arc::try_unwrap(Arc::clone(storage)) {
                Ok(storage) => storage,
                Err(shared_storage) => match shared_storage.as_ref() {
                    HoduStorage::CPU(cpu_storage) => HoduStorage::CPU(cpu_storage.clone()),
                    HoduStorage::Metal(metal_storage) => HoduStorage::Metal(metal_storage.clone()),
                },
            };
            let output_tensor = from_storage(output_storage, layout.clone(), false);
            Ok((output_name.clone(), output_tensor))
        })
        .collect::<HoduResult<Vec<_>>>()
}

#[cfg(not(all(feature = "std", feature = "rayon")))]
fn convert_storage_to_outputs(
    tensor_storage: &HashMap<TensorId, SharedStorage>,
    compiled: &HoduCompiledScript,
) -> HoduResult<Vec<(String, crate::tensor::Tensor)>> {
    let mut output_tensors = Vec::new();
    for (output_name, &tensor_id) in &compiled.output_mapping {
        if let Some(storage) = tensor_storage.get(&tensor_id) {
            if let Some(layout) = compiled.tensor_layouts.get(&tensor_id) {
                let output_storage = match Arc::try_unwrap(Arc::clone(storage)) {
                    Ok(storage) => storage,
                    Err(shared_storage) => match shared_storage.as_ref() {
                        HoduStorage::CPU(cpu_storage) => HoduStorage::CPU(cpu_storage.clone()),
                        HoduStorage::Metal(metal_storage) => HoduStorage::Metal(metal_storage.clone()),
                    },
                };
                let output_tensor = from_storage(output_storage, layout.clone(), false);
                output_tensors.push((output_name.clone(), output_tensor));
            } else {
                return Err(HoduError::InternalError(format!(
                    "Layout not found for output tensor {tensor_id:?}"
                )));
            }
        } else {
            return Err(HoduError::InternalError(format!(
                "Storage not found for output tensor {tensor_id:?}"
            )));
        }
    }
    Ok(output_tensors)
}
