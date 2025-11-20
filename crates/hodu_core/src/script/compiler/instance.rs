use super::{script::ScriptCompiler, types::*};
use crate::{
    error::{HoduError, HoduResult},
    script::builder::ir::Module,
    types::{Compiler, Device, Runtime},
};

/// Compiler trait - different backends implement this
pub trait CompilerT: Send + Sync {
    /// Get compiler type
    fn compiler_type(&self) -> Compiler;

    /// Get runtime type
    fn runtime(&self) -> Runtime;

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

/// Compiler instance - wrapper around ScriptCompiler
pub struct CompilerInstance {
    inner: ScriptCompiler,
}

impl CompilerInstance {
    /// Create a compiler based on Compiler, Runtime and device
    pub fn new(compiler: Compiler, runtime: Runtime, device: Device) -> HoduResult<Self> {
        if !runtime.is_supported(device) {
            return Err(HoduError::CompilationError(format!(
                "runtime {:?} does not support device {:?}",
                runtime, device
            )));
        }

        Ok(Self {
            inner: ScriptCompiler::new(compiler, runtime, device),
        })
    }
}

impl CompilerT for CompilerInstance {
    fn compiler_type(&self) -> Compiler {
        self.inner.compiler_type()
    }

    fn runtime(&self) -> Runtime {
        self.inner.runtime()
    }

    fn device(&self) -> Device {
        self.inner.device()
    }

    fn compile(&self, module: &Module, options: CompileOptions) -> HoduResult<CompiledModule> {
        self.inner.compile(module, options)
    }

    fn validate(&self, module: &Module) -> HoduResult<()> {
        self.inner.validate(module)
    }
}
