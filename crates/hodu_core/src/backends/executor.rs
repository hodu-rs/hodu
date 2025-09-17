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
