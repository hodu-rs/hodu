use super::{hodu::HoduExecutor, types::*};
use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    script::compiler::CompiledModule,
    types::{Compiler, Device},
};

#[cfg(feature = "xla")]
use super::xla::XlaExecutor;

/// Executor trait - different backends implement this
pub trait ExecutorT: Send + Sync {
    /// Get compiler type this executor is for
    fn compiler_type(&self) -> Compiler;

    /// Get target device
    fn device(&self) -> Device;

    /// Execute a compiled module with inputs
    fn execute(&self, compiled: &CompiledModule, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs>;

    /// Cleanup resources (optional)
    fn cleanup(&mut self) -> HoduResult<()> {
        Ok(())
    }
}

/// Executor instance enum to hold different executor implementations
pub enum ExecutorInstance {
    Hodu(HoduExecutor),
    #[cfg(feature = "xla")]
    Xla(XlaExecutor),
}

impl ExecutorInstance {
    /// Create a HODU executor for the given device
    pub fn hodu(device: Device) -> Self {
        ExecutorInstance::Hodu(HoduExecutor::new(device))
    }

    #[cfg(feature = "xla")]
    /// Create an XLA executor for the given device
    pub fn xla(device: Device) -> Self {
        ExecutorInstance::Xla(XlaExecutor::new(device))
    }

    /// Create an executor based on Compiler type and device
    pub fn new(compiler: Compiler, device: Device) -> HoduResult<Self> {
        if !compiler.is_supported(device) {
            return Err(HoduError::InternalError(format!(
                "Compiler {:?} does not support device {:?}",
                compiler, device
            )));
        }

        match compiler {
            Compiler::HODU => Ok(Self::hodu(device)),
            #[cfg(feature = "xla")]
            Compiler::XLA => Ok(Self::xla(device)),
        }
    }
}

impl ExecutorT for ExecutorInstance {
    fn compiler_type(&self) -> Compiler {
        match self {
            ExecutorInstance::Hodu(e) => e.compiler_type(),
            #[cfg(feature = "xla")]
            ExecutorInstance::Xla(e) => e.compiler_type(),
        }
    }

    fn device(&self) -> Device {
        match self {
            ExecutorInstance::Hodu(e) => e.device(),
            #[cfg(feature = "xla")]
            ExecutorInstance::Xla(e) => e.device(),
        }
    }

    fn execute(&self, compiled: &CompiledModule, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
        match self {
            ExecutorInstance::Hodu(e) => e.execute(compiled, inputs),
            #[cfg(feature = "xla")]
            ExecutorInstance::Xla(e) => e.execute(compiled, inputs),
        }
    }

    fn cleanup(&mut self) -> HoduResult<()> {
        match self {
            ExecutorInstance::Hodu(e) => e.cleanup(),
            #[cfg(feature = "xla")]
            ExecutorInstance::Xla(e) => e.cleanup(),
        }
    }
}
