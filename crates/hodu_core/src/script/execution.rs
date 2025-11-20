use super::{
    compilation::CompilationManager,
    config::ScriptConfig,
    input_manager::InputManager,
    runtime::{ExecutionInputs, RuntimeInstance, RuntimeT},
};
use crate::{compat::*, error::HoduResult, tensor::Tensor};

/// Execution manager - handles script execution and executor caching
pub(crate) struct ExecutionManager {
    runtime_instance: Option<RuntimeInstance>,
}

impl ExecutionManager {
    /// Create a new execution manager
    pub fn new() -> Self {
        Self { runtime_instance: None }
    }

    /// Execute the compiled module
    pub fn run(
        &mut self,
        config: &ScriptConfig,
        compilation: &CompilationManager,
        inputs: &InputManager,
    ) -> HoduResult<HashMap<String, Tensor>> {
        let compiled = compilation.get_compiled()?;

        // Create or reuse executor instance
        if self.runtime_instance.is_none() {
            self.runtime_instance = Some(RuntimeInstance::new(config.runtime_type, config.device)?);
        }

        let executor = self.runtime_instance.as_ref().unwrap();

        // Convert inputs to ExecutionInputs format
        let execution_inputs: ExecutionInputs<'_> = inputs.as_execution_inputs().collect();

        // Execute
        let outputs = executor.execute(compiled, execution_inputs)?;

        Ok(outputs)
    }

    /// Invalidate executor cache (called when config changes)
    pub fn invalidate(&mut self) {
        self.runtime_instance = None;
    }
}

impl Default for ExecutionManager {
    fn default() -> Self {
        Self::new()
    }
}
