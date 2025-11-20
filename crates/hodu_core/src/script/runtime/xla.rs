mod builder;
mod conversion;
mod helpers;
pub mod ops;

use super::{instance::RuntimeT, types::*};
use crate::{
    error::{HoduError, HoduResult},
    script::compiler::CompiledModule,
    types::{Device, Runtime},
};

/// XLA Runtime (feature-gated)
///
/// Note: This Runtime requires the `xla` feature and `hodu_xla` crate to be available.
/// For now, it provides a basic structure but requires full XLA integration.
#[derive(Debug)]
pub struct XlaRuntime {
    device: Device,
}

impl XlaRuntime {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl RuntimeT for XlaRuntime {
    fn runtime_type(&self) -> Runtime {
        Runtime::XLA
    }

    fn device(&self) -> Device {
        self.device
    }

    fn execute(&self, compiled: &CompiledModule, inputs: ExecutionInputs<'_>) -> HoduResult<ExecutionOutputs> {
        // Validate inputs
        for name in compiled.input_mapping.keys() {
            if !inputs.contains_key(name.as_str()) {
                return Err(HoduError::MissingInput(name.clone()));
            }
        }

        builder::build_and_execute_xla(self.device, compiled, &inputs)
    }
}
