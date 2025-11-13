use super::{Builder, BuilderState};
use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{Op, OpParams},
    tensor::{Tensor, TensorId},
    types::Layout,
};

use super::super::ir::{Attribute, ConstantData, Instruction, ValueId};

impl Builder {
    /// Add an input tensor to the graph
    pub fn add_input(&self, name: &'static str, tensor: Tensor) -> HoduResult<()> {
        self.with_state_mut(|s| s.graph_inputs.push((name, tensor)))
            .ok_or_else(|| HoduError::BuilderNotFound(self.get_name()))
    }

    /// Add an output tensor to the graph
    pub fn add_output(&self, name: &'static str, tensor: Tensor) -> HoduResult<()> {
        self.with_state_mut(|s| s.graph_outputs.push((name, tensor)))
            .ok_or_else(|| HoduError::BuilderNotFound(self.get_name()))
    }

    /// Set the outputs for the graph (replaces existing outputs)
    pub fn set_outputs(&self, names: &[&'static str], tensors: &[Tensor]) -> HoduResult<()> {
        if names.len() != tensors.len() {
            return Err(HoduError::InternalError(format!(
                "Names length ({}) must match tensors length ({})",
                names.len(),
                tensors.len()
            )));
        }
        let outputs: Vec<(&'static str, Tensor)> = names
            .iter()
            .zip(tensors.iter())
            .map(|(&name, tensor)| (name, tensor.clone()))
            .collect();
        self.with_state_mut(|s| s.graph_outputs = outputs)
            .ok_or_else(|| HoduError::BuilderNotFound(self.get_name()))
    }

    /// Add an operation to the computation graph
    pub fn add_operation(
        &self,
        op: Op,
        op_params: Option<OpParams>,
        inputs: Vec<TensorId>,
        outputs: Vec<TensorId>,
        _input_layouts: Vec<Layout>,
        _output_layouts: Vec<Layout>,
    ) -> HoduResult<()> {
        self.with_state_mut(|s| {
            s.ensure_function_and_block()?;
            let input_values = s.map_input_tensors(&inputs)?;
            let result_value = s.allocate_result_value(&outputs)?;
            let attributes = convert_op_params(op_params);
            s.add_instruction(result_value, op, input_values, attributes)?;

            // Keep output tensors alive until build completes
            for &tensor_id in &outputs {
                let tensor = crate::tensor::tensor_from_id(tensor_id);
                s.intermediate_tensors.push(tensor);
            }

            Ok(())
        })
        .ok_or_else(|| HoduError::InternalError(format!("Builder {} not found", self.get_name())))?
    }
}

impl BuilderState {
    /// Map input tensor IDs to value IDs, loading constants as needed
    pub(super) fn map_input_tensors(&mut self, inputs: &[TensorId]) -> HoduResult<Vec<ValueId>> {
        let mut input_values = Vec::with_capacity(inputs.len());

        for &tensor_id in inputs {
            // Check if already mapped
            if let Some(&value_id) = self.tensor_to_value.get(&tensor_id) {
                input_values.push(value_id);
                continue;
            }

            // Check if this is a constant that needs to be loaded
            if crate::tensor::get(tensor_id).is_some() {
                let tensor = crate::tensor::tensor_from_id(tensor_id);
                if tensor.has_storage() {
                    // This is a constant - add LoadConstant instruction
                    let value_id = self.allocate_value_id();
                    self.tensor_to_value.insert(tensor_id, value_id);

                    // Add constant data to module if not already present
                    self.ensure_constant_loaded(tensor_id, &tensor)?;

                    // Add LoadConstant instruction
                    self.add_load_constant_instruction(value_id, tensor_id)?;

                    input_values.push(value_id);
                    continue;
                }
            }

            // Not a constant, just allocate value_id
            let value_id = self.allocate_value_id();
            self.tensor_to_value.insert(tensor_id, value_id);
            input_values.push(value_id);
        }

        Ok(input_values)
    }

    /// Allocate a result value ID and map the first output tensor to it
    pub(super) fn allocate_result_value(&mut self, outputs: &[TensorId]) -> HoduResult<ValueId> {
        let result_value = self.allocate_value_id();

        // Store output tensor to value mapping (for first output)
        if let Some(&first_output) = outputs.first() {
            self.tensor_to_value.insert(first_output, result_value);
        }

        Ok(result_value)
    }

    /// Add a compute instruction to the current block
    pub(super) fn add_instruction(
        &mut self,
        result: ValueId,
        op: Op,
        inputs: Vec<ValueId>,
        attributes: HashMap<String, Attribute>,
    ) -> HoduResult<()> {
        let instruction = Instruction::Compute {
            result,
            op,
            inputs,
            attributes,
        };

        if let (Some(fn_name), Some(block_id)) = (&self.current_function, &self.current_block) {
            if let Some(function) = self.module.get_function_mut(fn_name) {
                if let Some(block) = function.get_block_mut(*block_id) {
                    block.add_instruction(instruction);
                    return Ok(());
                }
            }
        }

        Err(HoduError::InternalError(
            "No current function or block available".to_string(),
        ))
    }

    /// Allocate a new value ID
    fn allocate_value_id(&mut self) -> ValueId {
        let value_id = ValueId(self.value_counter);
        self.value_counter += 1;
        value_id
    }

    /// Ensure a constant is loaded into the module
    fn ensure_constant_loaded(&mut self, tensor_id: TensorId, tensor: &Tensor) -> HoduResult<()> {
        if !self.module.constants.contains_key(&tensor_id) {
            let layout = tensor.layout();
            let cpu_storage = tensor.with_storage(|storage| storage.to_cpu_storage()).ok();

            if let Some(cpu_storage) = cpu_storage {
                let constant = ConstantData {
                    tensor_id,
                    shape: layout.shape().clone(),
                    dtype: tensor.dtype(),
                    data: cpu_storage.to_bytes(),
                    compression: None,
                };
                self.module.add_constant(tensor_id, constant);
            }
        }
        Ok(())
    }

    /// Add a LoadConstant instruction to the current block
    fn add_load_constant_instruction(&mut self, result: ValueId, tensor_id: TensorId) -> HoduResult<()> {
        if let (Some(fn_name), Some(block_id)) = (&self.current_function, &self.current_block) {
            if let Some(function) = self.module.get_function_mut(fn_name) {
                if let Some(block) = function.get_block_mut(*block_id) {
                    block.add_instruction(Instruction::LoadConstant { result, tensor_id });
                    return Ok(());
                }
            }
        }

        Err(HoduError::InternalError(
            "No current function or block available".to_string(),
        ))
    }
}

/// Convert OpParams to a HashMap of attributes
fn convert_op_params(op_params: Option<OpParams>) -> HashMap<String, Attribute> {
    let mut attributes = HashMap::new();

    if let Some(params) = op_params {
        if let Some(scalar) = params.scalar {
            attributes.insert("scalar".to_string(), Attribute::Scalar(scalar));
        }
        if !params.scalars.is_empty() {
            attributes.insert("scalars".to_string(), Attribute::Scalars(params.scalars.clone()));
        }
        if !params.dims.is_empty() {
            attributes.insert("dims".to_string(), Attribute::Scalars(params.dims.clone()));
        }
        if let Some(keep_dim) = params.keep_dim {
            attributes.insert("keep_dim".to_string(), Attribute::Bool(keep_dim));
        }
        if let Some(output_index) = params.output_index {
            attributes.insert("output_index".to_string(), Attribute::Int(output_index as i32));
        }
        if let Some(dtype) = params.dtype {
            attributes.insert("dtype".to_string(), Attribute::DType(dtype));
        }
    }

    attributes
}
