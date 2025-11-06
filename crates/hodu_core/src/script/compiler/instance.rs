#[cfg(feature = "xla")]
use super::xla::XlaCompiler;
use super::{hodu::HoduCompiler, types::*};
use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    script::builder::ir::Module,
    types::{Compiler, Device},
};

/// Compiler trait - different backends implement this
pub trait CompilerT: Send + Sync {
    /// Get compiler type
    fn compiler_type(&self) -> Compiler;

    /// Get target device
    fn device(&self) -> Device;

    /// Compile a module to executable form
    fn compile(&self, module: &Module, options: CompileOptions) -> HoduResult<CompiledModule>;

    /// Validate that the module can be compiled
    fn validate(&self, module: &Module) -> HoduResult<()> {
        if module.functions.is_empty() {
            return Err(HoduError::MissingFunction("module has no functions".to_string()));
        }
        Ok(())
    }
}

/// Compiler instance enum to hold different compiler implementations
pub enum CompilerInstance {
    Hodu(HoduCompiler),
    #[cfg(feature = "xla")]
    Xla(XlaCompiler),
}

impl CompilerInstance {
    /// Create a HODU compiler for the given device
    pub fn hodu(device: Device) -> Self {
        CompilerInstance::Hodu(HoduCompiler::new(device))
    }

    #[cfg(feature = "xla")]
    /// Create an XLA compiler for the given device
    pub fn xla(device: Device) -> Self {
        CompilerInstance::Xla(XlaCompiler::new(device))
    }

    /// Create a compiler based on Compiler type and device
    pub fn new(compiler: Compiler, device: Device) -> HoduResult<Self> {
        if !compiler.is_supported(device) {
            return Err(HoduError::CompilationError(format!(
                "compiler {:?} does not support device {:?}",
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

impl CompilerT for CompilerInstance {
    fn compiler_type(&self) -> Compiler {
        match self {
            CompilerInstance::Hodu(c) => c.compiler_type(),
            #[cfg(feature = "xla")]
            CompilerInstance::Xla(c) => c.compiler_type(),
        }
    }

    fn device(&self) -> Device {
        match self {
            CompilerInstance::Hodu(c) => c.device(),
            #[cfg(feature = "xla")]
            CompilerInstance::Xla(c) => c.device(),
        }
    }

    fn compile(&self, module: &Module, options: CompileOptions) -> HoduResult<CompiledModule> {
        match self {
            CompilerInstance::Hodu(c) => c.compile(module, options),
            #[cfg(feature = "xla")]
            CompilerInstance::Xla(c) => c.compile(module, options),
        }
    }

    fn validate(&self, module: &Module) -> HoduResult<()> {
        match self {
            CompilerInstance::Hodu(c) => c.validate(module),
            #[cfg(feature = "xla")]
            CompilerInstance::Xla(c) => c.validate(module),
        }
    }
}
