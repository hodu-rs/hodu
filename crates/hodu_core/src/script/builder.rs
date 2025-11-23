pub mod codegen;
pub mod config;
pub mod validate;

use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    script::Script,
};
use codegen::CodeGenerator;
pub use config::{BuildConfig, TargetArch, TargetConfig, TargetEnv, TargetOS, TargetVendor};
use inkwell::context::Context;

/// Build output type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildType {
    /// Executable binary
    Binary,
    /// Shared library (.so, .dylib, .dll)
    Library,
}

/// Builder for AOT compilation
pub struct Builder<'a> {
    script: &'a Script,
    config: BuildConfig,
    build_type: BuildType,
    output_path: Option<String>,
}

impl<'a> Builder<'a> {
    /// Create a new builder for the given script
    pub fn new(script: &'a Script) -> Self {
        Self {
            script,
            config: BuildConfig::default(),
            build_type: BuildType::Library,
            output_path: None,
        }
    }

    /// Set the build configuration (execution + target)
    pub fn config(mut self, config: BuildConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the target platform configuration
    pub fn target(mut self, target: TargetConfig) -> Self {
        self.config.target = target;
        self
    }

    /// Set the target device
    pub fn device(mut self, device: crate::types::Device) -> Self {
        self.config.device = device;
        self
    }

    /// Set the target runtime
    pub fn runtime(mut self, runtime: crate::types::Runtime) -> Self {
        self.config.runtime = runtime;
        self
    }

    /// Set the build type (binary or library)
    pub fn build_type(mut self, build_type: BuildType) -> Self {
        self.build_type = build_type;
        self
    }

    /// Set the output path
    pub fn output_path(mut self, path: impl Into<String>) -> Self {
        self.output_path = Some(path.into());
        self
    }

    /// Build the script to native code
    pub fn build(self) -> HoduResult<()> {
        // Validate output path is set
        let _output_path = self
            .output_path
            .ok_or_else(|| HoduError::InvalidArgument("Output path must be set before building".into()))?;

        // Create LLVM context and code generator
        let context = Context::create();
        let mut codegen = CodeGenerator::new(&context, "hodu_module", self.config.device, self.config.runtime);

        // Generate LLVM IR from Snapshot
        codegen.generate(self.script.snapshot())?;

        // Get the LLVM module
        let _module = codegen.module();

        // TODO: Verify the module

        // TODO: Apply LLVM optimization passes

        // TODO: Generate target machine code using target triple

        // TODO: Write to output file (requires std feature)
        #[cfg(feature = "std")]
        {
            // TODO: Write LLVM IR to file
        }

        #[cfg(not(feature = "std"))]
        {
            return Err(HoduError::UnsupportedOperation(
                "File output not available in no_std environment".into(),
            ));
        }

        Ok(())
    }
}
