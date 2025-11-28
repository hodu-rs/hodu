//! Compiler plugin interface for AOT/JIT compilation

use crate::{CompiledArtifact, Device, HoduResult, OutputFormat, Script};
use std::path::Path;

/// Compiler plugin interface
///
/// A compiler transforms Scripts into executable artifacts or files.
/// Examples: LLVM, Metal Compiler, XLA Compiler
pub trait CompilerPlugin: Send + Sync {
    /// Plugin name (e.g., "llvm", "metal", "xla")
    fn name(&self) -> &str;

    /// Plugin version
    fn version(&self) -> &str;

    /// List of supported target devices
    fn supported_devices(&self) -> Vec<Device>;

    /// List of supported output formats for a given device
    fn supported_formats(&self, device: Device) -> Vec<OutputFormat>;

    /// Check if this compiler supports the given device
    fn supports_device(&self, device: Device) -> bool {
        self.supported_devices().contains(&device)
    }

    /// Check if this compiler supports the given output format for a device
    fn supports_format(&self, device: Device, format: OutputFormat) -> bool {
        self.supported_formats(device).contains(&format)
    }

    /// JIT compile a Script into an in-memory artifact
    fn compile(&self, script: &Script, device: Device) -> HoduResult<CompiledArtifact>;

    /// AOT build a Script to a file
    fn build(&self, script: &Script, device: Device, format: OutputFormat, path: &Path) -> HoduResult<()>;
}

/// Plugin information for discovery
#[repr(C)]
pub struct CompilerPluginInfo {
    pub name: &'static str,
    pub version: &'static str,
}

/// Plugin entry point function type
/// Each plugin must export: `extern "C" fn hodu_compiler_plugin_create() -> *mut dyn CompilerPlugin`
pub type CompilerPluginCreateFn = unsafe extern "C" fn() -> *mut dyn CompilerPlugin;

/// Plugin destroy function type
/// Each plugin must export: `extern "C" fn hodu_compiler_plugin_destroy(ptr: *mut dyn CompilerPlugin)`
pub type CompilerPluginDestroyFn = unsafe extern "C" fn(ptr: *mut dyn CompilerPlugin);
