mod execute;
mod ops;

use super::{
    instance::RuntimeT,
    types::{self, ExecutionInputs, ExecutionOutputs},
};
use crate::{
    error::HoduResult,
    script::compiler::CompiledModule,
    types::{Device, Runtime},
};

/// HODU nativeRuntime
#[derive(Debug)]
pub struct HoduRuntime {
    device: Device,
}

impl HoduRuntime {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl RuntimeT for HoduRuntime {
    fn runtime_type(&self) -> Runtime {
        Runtime::HODU
    }

    fn device(&self) -> Device {
        self.device
    }

    fn execute(&self, compiled: &CompiledModule, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
        execute::execute(compiled, inputs)
    }
}
