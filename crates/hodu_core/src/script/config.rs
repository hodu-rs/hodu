use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    types::{Device, Runtime},
};

/// Script configuration - manages runtime and device settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScriptConfig {
    pub runtime_type: Runtime,
    pub device: Device,
}

impl ScriptConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self {
            runtime_type: Runtime::default(),
            device: Device::CPU,
        }
    }

    /// Validate runtime and device compatibility
    pub fn validate(&self) -> HoduResult<()> {
        if !self.runtime_type.is_supported(self.device) {
            return Err(HoduError::CompilationError(format!(
                "runtime {:?} does not support device {:?}. available devices for {:?}: {}",
                self.runtime_type,
                self.device,
                self.runtime_type,
                match self.runtime_type {
                    Runtime::HODU => "CPU, Metal, CUDA",
                    #[cfg(feature = "xla")]
                    Runtime::XLA => "CPU, CUDA",
                    #[allow(unreachable_patterns)]
                    _ => "unknown",
                }
            )));
        }
        Ok(())
    }

    /// Set runtime type
    pub fn set_runtime(&mut self, runtime: Runtime) {
        self.runtime_type = runtime;
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
