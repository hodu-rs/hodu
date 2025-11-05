mod base;
mod hodu;
mod instance;
mod types;
#[cfg(feature = "xla")]
mod xla;

pub use hodu::HoduCompiler;
pub use instance::{CompilerInstance, CompilerT};
pub use types::{CompileOptions, CompiledInstruction, CompiledModule, OptimizationLevel};
#[cfg(feature = "xla")]
pub use xla::XlaCompiler;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Compiler, Device};

    #[test]
    fn test_compile_options_default() {
        let options = CompileOptions::default();
        assert_eq!(options.device, Device::CPU);
        assert_eq!(options.optimization_level, OptimizationLevel::Basic);
        assert!(!options.enable_profiling);
    }

    #[test]
    fn test_compiler_instance_creation() {
        let compiler = CompilerInstance::new(Compiler::HODU, Device::CPU).unwrap();
        assert_eq!(compiler.compiler_type(), Compiler::HODU);
        assert_eq!(compiler.device(), Device::CPU);
    }

    #[test]
    fn test_hodu_compiler_direct() {
        let compiler = CompilerInstance::hodu(Device::CPU);
        assert_eq!(compiler.compiler_type(), Compiler::HODU);
    }
}
