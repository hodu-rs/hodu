pub mod builder;
pub mod compiler;
pub mod executor;

use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    tensor::Tensor,
    types::{Compiler, Device},
};
use builder::ir::Module;
use compiler::{CompileOptions, CompiledModule, CompilerInstance, CompilerT};
use executor::{ExecutionInputs, ExecutorInstance, ExecutorT};

/// Script - executable module with compilation and execution capabilities
pub struct Script {
    module: Module,
    compiler_type: Compiler,
    device: Device,
    runtime_inputs: HashMap<String, Tensor>,
    compiled: Option<CompiledModule>,
}

impl fmt::Debug for Script {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Script")
            .field("module", &self.module.name)
            .field("compiler", &self.compiler_type)
            .field("device", &self.device)
            .field("runtime_inputs", &self.runtime_inputs.len())
            .field("compiled", &self.compiled.is_some())
            .finish()
    }
}

impl Script {
    /// Create a new script from a module
    pub fn new(module: Module) -> Self {
        Self {
            module,
            compiler_type: Compiler::default(),
            device: Device::CPU,
            runtime_inputs: HashMap::new(),
            compiled: None,
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
        if self.compiler_type != compiler {
            self.compiler_type = compiler;
            // Invalidate compilation
            self.compiled = None;
        }
    }

    /// Set target device
    pub fn set_device(&mut self, device: Device) {
        if self.device != device {
            self.device = device;
            // Invalidate compilation
            self.compiled = None;
        }
    }

    /// Get current compiler type
    pub fn compiler(&self) -> Compiler {
        self.compiler_type
    }

    /// Get current device
    pub fn device(&self) -> Device {
        self.device
    }

    /// Add an input tensor
    pub fn set_input(&mut self, name: &str, tensor: Tensor) {
        self.runtime_inputs.insert(name.to_string(), tensor);
    }

    /// Add an input tensor (builder pattern)
    pub fn with_input(mut self, name: &str, tensor: Tensor) -> Self {
        self.set_input(name, tensor);
        self
    }

    /// Clear all runtime inputs
    pub fn clear_inputs(&mut self) {
        self.runtime_inputs.clear();
    }

    /// Get current runtime inputs
    pub fn inputs(&self) -> &HashMap<String, Tensor> {
        &self.runtime_inputs
    }

    /// Compile the module
    pub fn compile(&mut self) -> HoduResult<()> {
        // Validate compiler and device compatibility
        if !self.compiler_type.is_supported(self.device) {
            return Err(HoduError::InternalError(format!(
                "Compiler {:?} does not support device {:?}. Available devices for {:?}: {}",
                self.compiler_type,
                self.device,
                self.compiler_type,
                match self.compiler_type {
                    Compiler::HODU => "CPU, Metal, CUDA",
                    #[cfg(feature = "xla")]
                    Compiler::XLA => "CPU, CUDA",
                    #[allow(unreachable_patterns)]
                    _ => "unknown",
                }
            )));
        }

        // Convert runtime_inputs to target device if needed
        let mut converted_inputs = HashMap::new();
        for (name, tensor) in &self.runtime_inputs {
            let converted = if tensor.device() != self.device {
                // Get CPU storage
                let cpu_storage = tensor.with_storage(|s| s.to_cpu_storage())?;
                // Create new storage on target device
                let new_storage = match self.device {
                    Device::CPU => crate::be::storage::BackendStorage::CPU(cpu_storage),
                    #[cfg(feature = "metal")]
                    Device::Metal => crate::be::storage::BackendStorage::Metal(
                        crate::be_metal::storage::MetalStorage::from_cpu_storage(&cpu_storage)?,
                    ),
                    #[allow(unreachable_patterns)]
                    _ => {
                        return Err(HoduError::InternalError(format!(
                            "Unsupported device: {:?}",
                            self.device
                        )))
                    },
                };
                let layout = tensor.layout();
                crate::tensor::from_storage(new_storage, layout, true, false)
            } else {
                *tensor
            };
            converted_inputs.insert(name.clone(), converted);
        }
        self.runtime_inputs = converted_inputs;

        // Create compiler instance
        let compiler = CompilerInstance::new(self.compiler_type, self.device)?;

        // Compile options
        let options = CompileOptions {
            device: self.device,
            ..Default::default()
        };

        // Compile the module
        let compiled = compiler.compile(&self.module, options)?;
        self.compiled = Some(compiled);

        Ok(())
    }

    /// Check if the script is compiled
    pub fn is_compiled(&self) -> bool {
        self.compiled.is_some()
    }

    /// Execute the compiled script
    pub fn run(&mut self) -> HoduResult<HashMap<String, Tensor>> {
        // Compile if not already compiled
        if !self.is_compiled() {
            self.compile()?;
        }

        let compiled = self
            .compiled
            .as_ref()
            .ok_or_else(|| HoduError::InternalError("Script not compiled".to_string()))?;

        // Create executor instance
        let executor = ExecutorInstance::new(self.compiler_type, self.device)?;

        // Convert runtime inputs to ExecutionInputs format
        let inputs: ExecutionInputs<'_> = self.runtime_inputs.iter().map(|(k, v)| (k.as_str(), *v)).collect();

        // Execute
        let outputs = executor.execute(compiled, inputs)?;

        Ok(outputs)
    }

    /// Save the module
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> HoduResult<()> {
        self.module.save_bc(path)
    }

    /// Load a script from file
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> HoduResult<Self> {
        let module = Module::load_bc(path)?;
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
