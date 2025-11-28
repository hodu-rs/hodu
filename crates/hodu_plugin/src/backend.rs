//! Backend plugin interface for compilation and execution

use crate::{Device, HoduResult, OutputFormat, Script, Tensor};
use hodu_compat::*;
use std::path::Path;

/// Compiled module handle (opaque type)
/// Each backend implementation provides its own CompiledModule
pub struct CompiledModule {
    inner: Box<dyn CompiledModuleInner>,
}

impl CompiledModule {
    pub fn new<T: CompiledModuleInner + 'static>(inner: T) -> Self {
        Self { inner: Box::new(inner) }
    }

    pub fn execute(&self, inputs: &[(&str, &Tensor)]) -> HoduResult<HashMap<String, Tensor>> {
        self.inner.execute(inputs)
    }
}

/// Inner trait for compiled module implementations
pub trait CompiledModuleInner: Send + Sync {
    fn execute(&self, inputs: &[(&str, &Tensor)]) -> HoduResult<HashMap<String, Tensor>>;
}

/// Backend plugin interface
///
/// A backend handles both compilation and execution of Scripts.
/// Examples: LLVM (CPU/CUDA/ROCm), Metal, XLA, ONNX Runtime, TVM
pub trait BackendPlugin: Send + Sync {
    /// Plugin name (e.g., "llvm", "metal", "xla")
    fn name(&self) -> &str;

    /// Plugin version
    fn version(&self) -> &str;

    /// List of supported devices
    fn supported_devices(&self) -> Vec<Device>;

    /// List of supported output formats for a given device
    fn supported_formats(&self, device: Device) -> Vec<OutputFormat>;

    /// Check if this backend supports the given device
    fn supports_device(&self, device: Device) -> bool {
        self.supported_devices().contains(&device)
    }

    /// Check if this backend supports the given output format for a device
    fn supports_format(&self, device: Device, format: OutputFormat) -> bool {
        self.supported_formats(device).contains(&format)
    }

    /// JIT compile a Script for execution (loads into memory)
    fn compile(&self, script: &Script, device: Device) -> HoduResult<CompiledModule>;

    /// AOT build a Script to a file
    fn build(&self, script: &Script, device: Device, format: OutputFormat, path: &Path) -> HoduResult<()>;
}

/// Plugin information for discovery
#[repr(C)]
pub struct BackendPluginInfo {
    pub name: &'static str,
    pub version: &'static str,
}

/// Plugin entry point function type (returns raw pointer for FFI safety)
/// Each plugin must export: `extern "C" fn hodu_backend_plugin_create() -> *mut ()`
/// The returned pointer should be cast to `Box<dyn BackendPlugin>` using `from_raw`
pub type BackendPluginCreateFn = unsafe extern "C" fn() -> *mut ();

/// Plugin destroy function type
/// Each plugin must export: `extern "C" fn hodu_backend_plugin_destroy(ptr: *mut ())`
pub type BackendPluginDestroyFn = unsafe extern "C" fn(ptr: *mut ());
