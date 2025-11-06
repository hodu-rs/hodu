use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    types::{Compiler, Device},
};

/// Script configuration - manages compiler and device settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScriptConfig {
    pub compiler_type: Compiler,
    pub device: Device,
}

impl ScriptConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self {
            compiler_type: Compiler::default(),
            device: Device::CPU,
        }
    }

    /// Validate compiler and device compatibility
    pub fn validate(&self) -> HoduResult<()> {
        if !self.compiler_type.is_supported(self.device) {
            return Err(HoduError::CompilationError(format!(
                "compiler {:?} does not support device {:?}. available devices for {:?}: {}",
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
        Ok(())
    }

    /// Set compiler type
    pub fn set_compiler(&mut self, compiler: Compiler) {
        self.compiler_type = compiler;
    }

    /// Set device
    pub fn set_device(&mut self, device: Device) {
        self.device = device;
    }
}

impl Default for ScriptConfig {
    fn default() -> Self {
        Self::new()
    }
}
