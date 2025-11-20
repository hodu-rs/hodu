pub mod builder;
pub mod compiler;
pub mod executor;

mod compilation;
mod config;
mod execution;
mod input_manager;
#[cfg(feature = "serde")]
mod io;

use crate::{
    compat::*,
    error::HoduResult,
    tensor::Tensor,
    types::{Compiler, Device},
};
use builder::ir::Module;
use compilation::CompilationManager;
use config::ScriptConfig;
use execution::ExecutionManager;
use input_manager::InputManager;

/// Script - executable module with compilation and execution capabilities
pub struct Script {
    module: Module,
    config: ScriptConfig,
    inputs: InputManager,
    compilation: CompilationManager,
    execution: ExecutionManager,
}

impl fmt::Debug for Script {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Script")
            .field("module", &self.module.name)
            .field("compiler", &self.config.compiler_type)
            .field("device", &self.config.device)
            .field("runtime_inputs", &self.inputs.len())
            .field("compiled", &self.compilation.is_compiled())
            .finish()
    }
}

impl Script {
    /// Create a new script from a module
    pub fn new(module: Module) -> Self {
        Self {
            module,
            config: ScriptConfig::new(),
            inputs: InputManager::new(),
            compilation: CompilationManager::new(),
            execution: ExecutionManager::new(),
        }
    }

    /// Get the module
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Get the module name
    pub fn name(&self) -> &str {
        &self.module.name
    }

    /// Set compiler type
    pub fn set_compiler(&mut self, compiler: Compiler) {
        if self.config.compiler_type != compiler {
            self.config.set_compiler(compiler);
            // Invalidate compilation and execution caches
            self.compilation.invalidate();
            self.execution.invalidate();
        }
    }

    /// Set target device
    pub fn set_device(&mut self, device: Device) {
        if self.config.device != device {
            self.config.set_device(device);
            // Invalidate compilation and execution caches
            self.compilation.invalidate();
            self.execution.invalidate();
        }
    }

    /// Get current compiler type
    pub fn compiler(&self) -> Compiler {
        self.config.compiler_type
    }

    /// Get current device
    pub fn device(&self) -> Device {
        self.config.device
    }

    /// Add an input tensor
    pub fn set_input(&mut self, name: &str, tensor: Tensor) {
        self.inputs.set(name, tensor);
    }

    /// Add an input tensor (builder pattern)
    pub fn with_input(mut self, name: &str, tensor: Tensor) -> Self {
        self.set_input(name, tensor);
        self
    }

    /// Clear all runtime inputs
    pub fn clear_inputs(&mut self) {
        self.inputs.clear();
    }

    /// Get current runtime inputs
    pub fn inputs(&self) -> &HashMap<String, Tensor> {
        self.inputs.get()
    }

    /// Compile the module
    pub fn compile(&mut self) -> HoduResult<()> {
        self.compilation.compile(&self.module, &self.config, &mut self.inputs)
    }

    /// Check if the script is compiled
    pub fn is_compiled(&self) -> bool {
        self.compilation.is_compiled()
    }

    /// Execute the compiled script
    pub fn run(&mut self) -> HoduResult<HashMap<String, Tensor>> {
        // Compile if not already compiled
        if !self.is_compiled() {
            self.compile()?;
        }

        self.execution.run(&self.config, &self.compilation, &self.inputs)
    }

    /// Serialize the module to bytes (no_std compatible)
    #[cfg(feature = "serde")]
    pub fn to_bytes(&self) -> HoduResult<Vec<u8>> {
        io::save_module_to_bytes(&self.module)
    }

    /// Deserialize a script from bytes (no_std compatible)
    #[cfg(feature = "serde")]
    pub fn from_bytes(bytes: &[u8]) -> HoduResult<Self> {
        let module = io::load_module_from_bytes(bytes)?;
        Ok(Self::new(module))
    }

    /// Save the module to file (requires std)
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> HoduResult<()> {
        io::save_module(&self.module, path)
    }

    /// Load a script from file (requires std)
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> HoduResult<Self> {
        let module = io::load_module(path)?;
        Ok(Self::new(module))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_script_creation() {
        let module = Module::new("test".to_string());
        let script = Script::new(module);
        assert_eq!(script.name(), "test");
        assert_eq!(script.compiler(), Compiler::HODU);
        assert_eq!(script.device(), Device::CPU);
        assert!(!script.is_compiled());
    }

    #[test]
    fn test_script_set_compiler() {
        let module = Module::new("test".to_string());
        let mut script = Script::new(module);

        script.set_compiler(Compiler::HODU);
        assert_eq!(script.compiler(), Compiler::HODU);
    }

    #[test]
    fn test_script_set_device() {
        let module = Module::new("test".to_string());
        let mut script = Script::new(module);

        script.set_device(Device::CPU);
        assert_eq!(script.device(), Device::CPU);
    }
}
