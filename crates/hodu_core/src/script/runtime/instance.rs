use super::{hodu::HoduRuntime, types::*};
use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    script::compiler::CompiledModule,
    types::{Device, Runtime},
};

#[cfg(feature = "xla")]
use super::xla::XlaRuntime;

/// Runtime trait - different backends implement this
pub trait RuntimeT: Send + Sync {
    /// Get runtime type this Runtime is for
    fn runtime_type(&self) -> Runtime;

    /// Get target device
    fn device(&self) -> Device;

    /// Execute a compiled module with inputs
    fn execute(&self, compiled: &CompiledModule, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs>;

    /// Cleanup resources (optional)
    fn cleanup(&mut self) -> HoduResult<()> {
        Ok(())
    }
}

/// Runtime instance enum to hold different Runtime implementations
pub enum RuntimeInstance {
    Hodu(HoduRuntime),
    #[cfg(feature = "xla")]
    Xla(XlaRuntime),
}

impl RuntimeInstance {
    /// Create a HODU Runtime for the given device
    pub fn hodu(device: Device) -> Self {
        RuntimeInstance::Hodu(HoduRuntime::new(device))
    }

    #[cfg(feature = "xla")]
    /// Create an XLA Runtime for the given device
    pub fn xla(device: Device) -> Self {
        RuntimeInstance::Xla(XlaRuntime::new(device))
    }

    /// Create an Runtime based on Compiler type and device
    pub fn new(runtime: Runtime, device: Device) -> HoduResult<Self> {
        if !runtime.is_supported(device) {
            return Err(HoduError::InternalError(format!(
                "Runtime {:?} does not support device {:?}",
                runtime, device
            )));
        }

        match runtime {
            Runtime::HODU => Ok(Self::hodu(device)),
            #[cfg(feature = "xla")]
            Runtime::XLA => Ok(Self::xla(device)),
        }
    }
}

impl RuntimeT for RuntimeInstance {
    fn runtime_type(&self) -> Runtime {
        match self {
            RuntimeInstance::Hodu(e) => e.runtime_type(),
            #[cfg(feature = "xla")]
            RuntimeInstance::Xla(e) => e.runtime_type(),
        }
    }

    fn device(&self) -> Device {
        match self {
            RuntimeInstance::Hodu(e) => e.device(),
            #[cfg(feature = "xla")]
            RuntimeInstance::Xla(e) => e.device(),
        }
    }

    fn execute(&self, compiled: &CompiledModule, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
        match self {
            RuntimeInstance::Hodu(e) => e.execute(compiled, inputs),
            #[cfg(feature = "xla")]
            RuntimeInstance::Xla(e) => e.execute(compiled, inputs),
        }
    }

    fn cleanup(&mut self) -> HoduResult<()> {
        match self {
            RuntimeInstance::Hodu(e) => e.cleanup(),
            #[cfg(feature = "xla")]
            RuntimeInstance::Xla(e) => e.cleanup(),
        }
    }
}
