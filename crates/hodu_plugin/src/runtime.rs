//! Runtime plugin interface for executing compiled artifacts

use crate::{CompiledArtifact, Device, HoduResult, OutputFormat};
use hodu_compat::*;
use hodu_core::types::DType;
use std::path::Path;

/// Raw tensor data for cross-plugin communication
///
/// This struct is used to pass tensor data between the main binary and plugins
/// without depending on the Tensor registry.
#[derive(Debug, Clone)]
pub struct TensorData {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl TensorData {
    pub fn new(data: Vec<u8>, shape: Vec<usize>, dtype: DType) -> Self {
        Self { data, shape, dtype }
    }

    /// Number of elements
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Executable module loaded into a runtime
///
/// This is an opaque handle to a loaded module that can be executed.
pub struct ExecutableModule {
    inner: Box<dyn ExecutableModuleInner>,
}

impl ExecutableModule {
    pub fn new<T: ExecutableModuleInner + 'static>(inner: T) -> Self {
        Self { inner: Box::new(inner) }
    }

    /// Execute with named inputs (raw tensor data)
    pub fn execute(&self, inputs: &[(&str, TensorData)]) -> HoduResult<HashMap<String, TensorData>> {
        self.inner.execute(inputs)
    }
}

/// Inner trait for executable module implementations
pub trait ExecutableModuleInner: Send + Sync {
    fn execute(&self, inputs: &[(&str, TensorData)]) -> HoduResult<HashMap<String, TensorData>>;
}

/// Runtime plugin interface
///
/// A runtime loads and executes compiled artifacts.
/// Examples: Native Runtime (dlopen), CUDA Runtime, Metal Runtime, ONNX Runtime
pub trait RuntimePlugin: Send + Sync {
    /// Plugin name (e.g., "native", "cuda", "metal", "onnxruntime")
    fn name(&self) -> &str;

    /// Plugin version
    fn version(&self) -> &str;

    /// List of supported devices
    fn supported_devices(&self) -> Vec<Device>;

    /// List of loadable formats for a given device
    fn loadable_formats(&self, device: Device) -> Vec<OutputFormat>;

    /// Check if this runtime supports the given device
    fn supports_device(&self, device: Device) -> bool {
        self.supported_devices().contains(&device)
    }

    /// Check if this runtime can load the given format
    fn can_load_format(&self, device: Device, format: OutputFormat) -> bool {
        self.loadable_formats(device).contains(&format)
    }

    /// Load from an in-memory compiled artifact
    fn load(&self, artifact: &CompiledArtifact, device: Device) -> HoduResult<ExecutableModule>;

    /// Load from a file (AOT compiled artifact)
    fn load_file(&self, path: &Path, device: Device) -> HoduResult<ExecutableModule>;
}

/// Plugin information for discovery
#[repr(C)]
pub struct RuntimePluginInfo {
    pub name: &'static str,
    pub version: &'static str,
}

/// Plugin entry point function type
/// Each plugin must export: `extern "C" fn hodu_runtime_plugin_create() -> *mut dyn RuntimePlugin`
pub type RuntimePluginCreateFn = unsafe extern "C" fn() -> *mut dyn RuntimePlugin;

/// Plugin destroy function type
/// Each plugin must export: `extern "C" fn hodu_runtime_plugin_destroy(ptr: *mut dyn RuntimePlugin)`
pub type RuntimePluginDestroyFn = unsafe extern "C" fn(ptr: *mut dyn RuntimePlugin);
