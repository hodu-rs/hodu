use super::{
    builder::ir::Module,
    compiler::{CompileOptions, CompiledModule, CompilerInstance, CompilerT},
    config::ScriptConfig,
    input_manager::InputManager,
};
use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
};

/// Compilation manager - handles module compilation and caching
pub(crate) struct CompilationManager {
    compiled: Option<CompiledModule>,
    compiler_instance: Option<CompilerInstance>,
}

impl CompilationManager {
    /// Create a new compilation manager
    pub fn new() -> Self {
        Self {
            compiled: None,
            compiler_instance: None,
        }
    }

    /// Compile a module with given configuration
    pub fn compile(&mut self, module: &Module, config: &ScriptConfig, inputs: &mut InputManager) -> HoduResult<()> {
        // Validate compiler and device compatibility
        config.validate()?;

        // Convert inputs to target device
        inputs.convert_to_device(config.device)?;

        // Create or reuse compiler instance
        if self.compiler_instance.is_none() {
            self.compiler_instance = Some(CompilerInstance::new(config.compiler_type, config.device)?);
        }

        let compiler = self
            .compiler_instance
            .as_ref()
            .ok_or_else(|| HoduError::InternalError("Compiler instance not initialized".to_string()))?;

        // Compile options
        let options = CompileOptions {
            device: config.device,
            ..Default::default()
        };

        // Compile the module
        let compiled = compiler.compile(module, options)?;
        self.compiled = Some(compiled);

        Ok(())
    }

    /// Check if module is compiled
    pub fn is_compiled(&self) -> bool {
        self.compiled.is_some()
    }

    /// Get compiled module reference
    pub fn get_compiled(&self) -> HoduResult<&CompiledModule> {
        self.compiled
            .as_ref()
            .ok_or_else(|| HoduError::InternalError("Module not compiled".to_string()))
    }

    /// Invalidate compilation cache (called when config changes)
    pub fn invalidate(&mut self) {
        self.compiled = None;
        self.compiler_instance = None;
    }
}

impl Default for CompilationManager {
    fn default() -> Self {
        Self::new()
    }
}
