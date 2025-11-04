mod hodu;
mod instance;
mod types;
#[cfg(feature = "xla")]
mod xla;

pub use hodu::HoduExecutor;
pub use instance::{ExecutorInstance, ExecutorT};
pub use types::{ExecutionInputs, ExecutionOutputs};
#[cfg(feature = "xla")]
pub use xla::XlaExecutor;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Compiler, Device};

    #[test]
    fn test_executor_instance_creation() {
        let executor = ExecutorInstance::new(Compiler::HODU, Device::CPU).unwrap();
        assert_eq!(executor.compiler_type(), Compiler::HODU);
        assert_eq!(executor.device(), Device::CPU);
    }

    #[test]
    fn test_hodu_executor_direct() {
        let executor = ExecutorInstance::hodu(Device::CPU);
        assert_eq!(executor.compiler_type(), Compiler::HODU);
    }
}
