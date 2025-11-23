pub mod config;
pub mod validate;

use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    script::Script,
};
pub use config::{BuildConfig, ExecutionConfig, TargetArch, TargetConfig, TargetEnv, TargetOS, TargetVendor};

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
    target: TargetConfig,
    build_type: BuildType,
    output_path: Option<String>,
}

impl<'a> Builder<'a> {
    /// Create a new builder for the given script
    pub fn new(script: &'a Script) -> Self {
        Self {
            script,
            target: TargetConfig::default(),
            build_type: BuildType::Library,
            output_path: None,
        }
    }

    /// Set the target platform configuration
    pub fn target(mut self, target: TargetConfig) -> Self {
        self.target = target;
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

        // TODO: Validate snapshot for target platform

        // TODO: Generate LLVM IR or similar from Snapshot

        // TODO: Compile to native code using target triple

        // TODO: Link and produce output file

        // Placeholder for now
        Err(HoduError::NotImplemented("AOT compilation not yet implemented".into()))
    }
}
