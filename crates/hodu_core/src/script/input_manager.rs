use crate::{error::HoduResult, layer::compat::*, tensor::Tensor, types::Device};

/// Input manager - handles runtime input tensors
#[derive(Debug, Clone)]
pub(crate) struct InputManager {
    runtime_inputs: HashMap<String, Tensor>,
}

impl InputManager {
    /// Create a new empty input manager
    pub fn new() -> Self {
        Self {
            runtime_inputs: HashMap::new(),
        }
    }

    /// Set an input tensor
    pub fn set(&mut self, name: &str, tensor: Tensor) {
        self.runtime_inputs.insert(name.to_string(), tensor);
    }

    /// Clear all inputs
    pub fn clear(&mut self) {
        self.runtime_inputs.clear();
    }

    /// Get reference to inputs
    pub fn get(&self) -> &HashMap<String, Tensor> {
        &self.runtime_inputs
    }

    /// Get number of inputs
    pub fn len(&self) -> usize {
        self.runtime_inputs.len()
    }

    /// Convert all inputs to target device
    pub fn convert_to_device(&mut self, device: Device) -> HoduResult<()> {
        let mut converted = HashMap::new();
        for (name, tensor) in &self.runtime_inputs {
            let converted_tensor = tensor.to_device(device)?;
            converted.insert(name.clone(), converted_tensor);
        }
        self.runtime_inputs = converted;
        Ok(())
    }

    /// Get inputs as iterator for execution
    pub fn as_execution_inputs(&self) -> impl Iterator<Item = (&str, Tensor)> + '_ {
        self.runtime_inputs.iter().map(|(k, v)| (k.as_str(), v.clone()))
    }
}

impl Default for InputManager {
    fn default() -> Self {
        Self::new()
    }
}
