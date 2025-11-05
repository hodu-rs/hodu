mod execute;
mod ops;

use super::{
    instance::ExecutorT,
    types::{self, ExecutionInputs, ExecutionOutputs},
};
use crate::{
    error::HoduResult,
    script::compiler::CompiledModule,
    types::{Compiler, Device},
};

/// HODU native executor
#[derive(Debug)]
pub struct HoduExecutor {
    device: Device,
}

impl HoduExecutor {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl ExecutorT for HoduExecutor {
    fn compiler_type(&self) -> Compiler {
        Compiler::HODU
    }

    fn device(&self) -> Device {
        self.device
    }

    fn execute(&self, compiled: &CompiledModule, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
        execute::execute(self.device, compiled, inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hodu_executor_creation() {
        let executor = HoduExecutor::new(Device::CPU);
        assert_eq!(executor.compiler_type(), Compiler::HODU);
        assert_eq!(executor.device(), Device::CPU);
    }
}
