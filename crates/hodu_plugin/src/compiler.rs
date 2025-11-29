//! Compiler plugin interface for AOT/JIT compilation

use crate::{CompiledArtifact, Device, HoduResult, OutputFormat, Snapshot};
use std::path::Path;

/// Compiler plugin interface
///
/// A compiler transforms Snapshots into executable artifacts or files.
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

    /// JIT compile a Snapshot into an in-memory artifact
    fn compile(&self, snapshot: &Snapshot, device: Device) -> HoduResult<CompiledArtifact>;

    /// AOT build a Snapshot to a file
    fn build(&self, snapshot: &Snapshot, device: Device, format: OutputFormat, path: &Path) -> HoduResult<()>;
}

/// Plugin information for discovery
#[repr(C)]
pub struct CompilerPluginInfo {
    pub name: &'static str,
    pub version: &'static str,
}

/// Opaque plugin handle for FFI
///
/// This wraps a `Box<dyn CompilerPlugin>` but is represented as a raw pointer
/// for FFI safety. The actual type is erased at the C boundary.
#[repr(C)]
pub struct CompilerPluginHandle {
    _opaque: [u8; 0],
}

/// Plugin entry point function type
/// Each plugin must export: `extern "C" fn hodu_compiler_plugin_create() -> *mut CompilerPluginHandle`
pub type CompilerPluginCreateFn = unsafe extern "C" fn() -> *mut CompilerPluginHandle;

/// Plugin destroy function type
/// Each plugin must export: `extern "C" fn hodu_compiler_plugin_destroy(ptr: *mut CompilerPluginHandle)`
pub type CompilerPluginDestroyFn = unsafe extern "C" fn(ptr: *mut CompilerPluginHandle);

impl CompilerPluginHandle {
    /// Create a handle from a boxed plugin (called from plugin side)
    pub fn from_boxed(plugin: Box<dyn CompilerPlugin>) -> *mut Self {
        Box::into_raw(Box::new(plugin)) as *mut Self
    }

    /// Convert handle back to boxed plugin (called from host side)
    ///
    /// # Safety
    /// The handle must have been created by `from_boxed` and not yet destroyed
    pub unsafe fn into_boxed(ptr: *mut Self) -> Box<Box<dyn CompilerPlugin>> {
        Box::from_raw(ptr as *mut Box<dyn CompilerPlugin>)
    }

    /// Get a reference to the plugin
    ///
    /// # Safety
    /// The handle must be valid
    pub unsafe fn as_ref<'a>(ptr: *mut Self) -> &'a dyn CompilerPlugin {
        let boxed = &*(ptr as *mut Box<dyn CompilerPlugin>);
        boxed.as_ref()
    }
}
