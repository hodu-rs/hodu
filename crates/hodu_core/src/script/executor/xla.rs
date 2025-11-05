mod builder;
mod conversion;
mod helpers;
pub mod ops;

use super::{instance::ExecutorT, types::*};
use crate::{
    error::{HoduError, HoduResult},
    script::compiler::CompiledModule,
    types::{Compiler, Device},
};

/// XLA executor (feature-gated)
///
/// Note: This executor requires the `xla` feature and `hodu_xla` crate to be available.
/// For now, it provides a basic structure but requires full XLA integration.
#[derive(Debug)]
pub struct XlaExecutor {
    device: Device,
}

impl XlaExecutor {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl ExecutorT for XlaExecutor {
    fn compiler_type(&self) -> Compiler {
        Compiler::XLA
    }

    fn device(&self) -> Device {
        self.device
    }

    fn execute(&self, compiled: &CompiledModule, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
        // Validate inputs
        for name in compiled.input_mapping.keys() {
            if !inputs.contains_key(name.as_str()) {
                return Err(HoduError::InternalError(format!("Missing required input: {}", name)));
            }
        }

        builder::build_and_execute_xla(self.device, compiled, &inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xla_executor_creation() {
        let executor = XlaExecutor::new(Device::CPU);
        assert_eq!(executor.compiler_type(), Compiler::XLA);
        assert_eq!(executor.device(), Device::CPU);
    }
}
