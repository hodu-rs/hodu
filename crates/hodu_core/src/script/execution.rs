use super::{
    compilation::CompilationManager,
    config::ScriptConfig,
    executor::{ExecutionInputs, ExecutorInstance, ExecutorT},
    input_manager::InputManager,
};
use crate::{compat::*, error::HoduResult, tensor::Tensor};

/// Execution manager - handles script execution and executor caching
pub(crate) struct ExecutionManager {
    executor_instance: Option<ExecutorInstance>,
}

impl ExecutionManager {
    /// Create a new execution manager
    pub fn new() -> Self {
        Self {
            executor_instance: None,
        }
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
        if self.executor_instance.is_none() {
            self.executor_instance = Some(ExecutorInstance::new(config.compiler_type, config.device)?);
        }

        let executor = self.executor_instance.as_ref().unwrap();

        // Convert inputs to ExecutionInputs format
        let execution_inputs: ExecutionInputs<'_> = inputs.as_execution_inputs().collect();

        // Execute
        let outputs = executor.execute(compiled, execution_inputs)?;

        Ok(outputs)
    }

    /// Invalidate executor cache (called when config changes)
    pub fn invalidate(&mut self) {
        self.executor_instance = None;
    }
}

impl Default for ExecutionManager {
    fn default() -> Self {
        Self::new()
    }
}
