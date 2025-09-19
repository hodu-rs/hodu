#[cfg(feature = "xla")]
use crate::error::HoduError;
use crate::{backends::script::Script, compat::*, error::HoduResult, tensor::Tensor, types::device::Device};

pub trait ExecutorT: Send + Sync {
    type CompiledScript;

    fn backend_name(&self) -> &'static str;
    fn supported_devices(&self) -> Vec<Device>;
    fn current_device(&self) -> Device;

    fn compile(&mut self, script: &Script, options: CompileOptions) -> HoduResult<Self::CompiledScript>;

    fn execute(&self, compiled: &Self::CompiledScript, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs>;

    fn cleanup(&mut self) -> HoduResult<()>;
}

#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub optimization_level: OptimizationLevel,
    pub target_device: Device,
    pub memory_limit: Option<usize>,
    pub enable_profiling: bool,
    pub debug_mode: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Basic,
            target_device: Device::CPU,
            memory_limit: None,
            enable_profiling: false,
            debug_mode: false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
}

pub type ExecutionInputs<'a> = HashMap<&'a str, Tensor>;
pub type ExecutionOutputs = HashMap<String, Tensor>;

use crate::backends::be_hodu::executor::HoduExecutor;
#[cfg(feature = "xla")]
use crate::backends::be_xla::executor::XlaExecutor;

pub enum Executor {
    Hodu(HoduExecutor),
    #[cfg(feature = "xla")]
    Xla(XlaExecutor),
}

pub enum CompiledScript {
    Hodu(<HoduExecutor as ExecutorT>::CompiledScript),
    #[cfg(feature = "xla")]
    Xla(<XlaExecutor as ExecutorT>::CompiledScript),
}

impl ExecutorT for Executor {
    type CompiledScript = CompiledScript;

    fn backend_name(&self) -> &'static str {
        match self {
            Executor::Hodu(e) => e.backend_name(),
            #[cfg(feature = "xla")]
            Executor::Xla(e) => e.backend_name(),
        }
    }

    fn supported_devices(&self) -> Vec<Device> {
        match self {
            Executor::Hodu(e) => e.supported_devices(),
            #[cfg(feature = "xla")]
            Executor::Xla(e) => e.supported_devices(),
        }
    }

    fn current_device(&self) -> Device {
        match self {
            Executor::Hodu(e) => e.current_device(),
            #[cfg(feature = "xla")]
            Executor::Xla(e) => e.current_device(),
        }
    }

    fn compile(&mut self, script: &Script, options: CompileOptions) -> HoduResult<Self::CompiledScript> {
        match self {
            Executor::Hodu(e) => e.compile(script, options).map(CompiledScript::Hodu),
            #[cfg(feature = "xla")]
            Executor::Xla(e) => e.compile(script, options).map(CompiledScript::Xla),
        }
    }

    fn execute(&self, compiled: &Self::CompiledScript, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
        match (self, compiled) {
            (Executor::Hodu(e), CompiledScript::Hodu(c)) => e.execute(c, inputs),
            #[cfg(feature = "xla")]
            (Executor::Xla(e), CompiledScript::Xla(c)) => e.execute(c, inputs),
            #[cfg(feature = "xla")]
            _ => Err(HoduError::InternalError(
                "Mismatched executor and compiled script".to_string(),
            )),
        }
    }

    fn cleanup(&mut self) -> HoduResult<()> {
        match self {
            Executor::Hodu(e) => e.cleanup(),
            #[cfg(feature = "xla")]
            Executor::Xla(e) => e.cleanup(),
        }
    }
}
