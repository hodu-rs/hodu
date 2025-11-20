pub mod builder;
pub mod compiler;
pub mod runtime;

mod compilation;
mod config;
mod execution;
mod input_manager;
#[cfg(feature = "serde")]
mod io;
mod op_params;

use crate::{
    compat::*,
    error::HoduResult,
    tensor::Tensor,
    types::{Device, Runtime},
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
            .field("runtime", &self.config.runtime_type)
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

    /// Set runtime type
    pub fn set_runtime(&mut self, runtime: Runtime) {
        if self.config.runtime_type != runtime {
            self.config.set_runtime(runtime);
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

    /// Get current runtime type
    pub fn runtime(&self) -> Runtime {
        self.config.runtime_type
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
