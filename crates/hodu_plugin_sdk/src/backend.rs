//! Backend plugin interface for model execution and AOT compilation
//!
//! A `BackendPlugin` combines Runner (execution) and Builder (AOT compilation) capabilities.
//! Plugins can implement one or both functionalities based on their `BackendCapabilities`.

use crate::{BuildFormat, HoduResult, Snapshot, TensorData};
use hodu_compat::*;
use hodu_core::types::Device;
use std::path::Path;

/// Capabilities flags for backend plugins
#[derive(Debug, Clone, Copy, Default)]
pub struct BackendCapabilities {
    /// Whether this plugin supports running models (`hodu run`)
    pub runner: bool,
    /// Whether this plugin supports building artifacts (`hodu build`)
    pub builder: bool,
}

impl BackendCapabilities {
    /// Create capabilities with both runner and builder
    pub fn full() -> Self {
        Self {
            runner: true,
            builder: true,
        }
    }

    /// Create runner-only capabilities
    pub fn runner_only() -> Self {
        Self {
            runner: true,
            builder: false,
        }
    }

    /// Create builder-only capabilities
    pub fn builder_only() -> Self {
        Self {
            runner: false,
            builder: true,
        }
    }
}

/// Build target specification for AOT compilation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BuildTarget {
    /// Target triple (e.g., "x86_64-unknown-linux-gnu", "aarch64-apple-darwin")
    pub triple: String,
    /// Target device (CPU, Metal, CUDA)
    pub device: Device,
}

impl BuildTarget {
    /// Create a new build target
    pub fn new(triple: impl Into<String>, device: Device) -> Self {
        Self {
            triple: triple.into(),
            device,
        }
    }

    /// Create a build target for the current host system
    pub fn host(device: Device) -> Self {
        Self::new(env!("HOST_TARGET"), device)
    }
}

/// Backend plugin interface
///
/// Combines Runner (model execution) and Builder (AOT compilation) capabilities.
/// Plugins can implement one or both based on their `capabilities()`.
///
/// # Examples
///
/// ```ignore
/// use hodu_plugin_sdk::*;
///
/// pub struct MyBackend;
///
/// impl BackendPlugin for MyBackend {
///     fn name(&self) -> &str { "my-backend" }
///     fn version(&self) -> &str { "0.1.0" }
///
///     fn capabilities(&self) -> BackendCapabilities {
///         BackendCapabilities::runner_only()
///     }
///
///     // Runner implementation
///     fn supported_devices(&self) -> Vec<Device> {
///         vec![Device::CPU]
///     }
///
///     fn run(&self, snapshot: &Snapshot, device: Device, inputs: &[(&str, TensorData)])
///         -> HoduResult<HashMap<String, TensorData>> {
///         // ... implementation
///     }
///
///     // Builder - return empty since we only support runner
///     fn supported_targets(&self) -> Vec<BuildTarget> { vec![] }
///     fn supported_formats(&self, _target: &BuildTarget) -> Vec<OutputFormat> { vec![] }
///     fn build(&self, _: &Snapshot, _: &BuildTarget, _: OutputFormat, _: &Path)
///         -> HoduResult<()> {
///         Err(HoduError::BackendError("build not supported".into()))
///     }
/// }
/// ```
pub trait BackendPlugin: Send + Sync {
    /// Plugin name (e.g., "cpu", "metal", "cuda", "llvm")
    fn name(&self) -> &str;

    /// Plugin version (semver)
    fn version(&self) -> &str;

    /// Capabilities provided by this plugin
    fn capabilities(&self) -> BackendCapabilities;

    // === Runner functionality (`hodu run`) ===

    /// Devices supported for running models
    fn supported_devices(&self) -> Vec<Device>;

    /// Run a model on the specified device
    ///
    /// # Arguments
    /// * `snapshot` - The model to execute
    /// * `device` - Target device (must be in `supported_devices()`)
    /// * `inputs` - Named input tensors
    ///
    /// # Returns
    /// Named output tensors
    fn run(
        &self,
        snapshot: &Snapshot,
        device: Device,
        inputs: &[(&str, TensorData)],
    ) -> HoduResult<HashMap<String, TensorData>>;

    // === Builder functionality (`hodu build`) ===

    /// Build targets supported for AOT compilation
    fn supported_targets(&self) -> Vec<BuildTarget>;

    /// Output formats supported for a given build target
    fn supported_formats(&self, target: &BuildTarget) -> Vec<BuildFormat>;

    /// Build an AOT artifact
    ///
    /// # Arguments
    /// * `snapshot` - The model to compile
    /// * `target` - Target specification (triple + device)
    /// * `format` - Output format (must be in `supported_formats(target)`)
    /// * `output` - Output file path
    fn build(&self, snapshot: &Snapshot, target: &BuildTarget, format: BuildFormat, output: &Path) -> HoduResult<()>;

    // === Convenience methods ===

    /// Check if this plugin supports a device for running
    fn supports_device(&self, device: Device) -> bool {
        self.capabilities().runner && self.supported_devices().contains(&device)
    }

    /// Check if this plugin supports a build target
    fn supports_target(&self, target: &BuildTarget) -> bool {
        self.capabilities().builder && self.supported_targets().contains(target)
    }
}

/// Plugin metadata for SDK version verification
#[repr(C)]
pub struct BackendPluginMetadata {
    /// SDK version used to compile this plugin (semver)
    pub sdk_version: *const std::ffi::c_char,
    /// Plugin name
    pub name: *const std::ffi::c_char,
    /// Plugin version
    pub version: *const std::ffi::c_char,
}

/// Opaque plugin handle for FFI
///
/// This wraps a `Box<dyn BackendPlugin>` but is represented as a raw pointer
/// for FFI safety. The actual type is erased at the C boundary.
#[repr(C)]
pub struct BackendPluginHandle {
    _opaque: [u8; 0],
}

/// Plugin entry point function type
/// Each plugin must export: `extern "C" fn hodu_backend_plugin_create() -> *mut BackendPluginHandle`
pub type BackendPluginCreateFn = unsafe extern "C" fn() -> *mut BackendPluginHandle;

/// Plugin destroy function type
/// Each plugin must export: `extern "C" fn hodu_backend_plugin_destroy(ptr: *mut BackendPluginHandle)`
pub type BackendPluginDestroyFn = unsafe extern "C" fn(ptr: *mut BackendPluginHandle);

/// Plugin metadata function type
/// Each plugin must export: `extern "C" fn hodu_backend_plugin_metadata() -> BackendPluginMetadata`
pub type BackendPluginMetadataFn = unsafe extern "C" fn() -> BackendPluginMetadata;

impl BackendPluginHandle {
    /// Create a handle from a boxed plugin (called from plugin side)
    pub fn from_boxed(plugin: Box<dyn BackendPlugin>) -> *mut Self {
        Box::into_raw(Box::new(plugin)) as *mut Self
    }

    /// Convert handle back to boxed plugin (called from host side)
    ///
    /// # Safety
    /// The handle must have been created by `from_boxed` and not yet destroyed
    pub unsafe fn into_boxed(ptr: *mut Self) -> Box<Box<dyn BackendPlugin>> {
        Box::from_raw(ptr as *mut Box<dyn BackendPlugin>)
    }

    /// Get a reference to the plugin
    ///
    /// # Safety
    /// The handle must be valid
    pub unsafe fn as_ref<'a>(ptr: *mut Self) -> &'a dyn BackendPlugin {
        let boxed = &*(ptr as *mut Box<dyn BackendPlugin>);
        boxed.as_ref()
    }
}

/// Macro to export backend plugin FFI functions
///
/// # Example
///
/// ```ignore
/// use hodu_plugin_sdk::*;
///
/// #[derive(Default)]
/// pub struct MyBackend;
/// impl BackendPlugin for MyBackend { /* ... */ }
///
/// export_backend_plugin!(MyBackend, "my-backend", "0.1.0");
/// ```
#[macro_export]
macro_rules! export_backend_plugin {
    ($plugin_type:ty, $name:literal, $version:literal) => {
        #[no_mangle]
        pub extern "C" fn hodu_backend_plugin_create() -> *mut $crate::BackendPluginHandle {
            let boxed: Box<dyn $crate::BackendPlugin> = Box::new(<$plugin_type>::default());
            $crate::BackendPluginHandle::from_boxed(boxed)
        }

        #[no_mangle]
        pub unsafe extern "C" fn hodu_backend_plugin_destroy(ptr: *mut $crate::BackendPluginHandle) {
            if !ptr.is_null() {
                drop($crate::BackendPluginHandle::into_boxed(ptr));
            }
        }

        #[no_mangle]
        pub extern "C" fn hodu_backend_plugin_metadata() -> $crate::BackendPluginMetadata {
            $crate::BackendPluginMetadata {
                sdk_version: concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast(),
                name: concat!($name, "\0").as_ptr().cast(),
                version: concat!($version, "\0").as_ptr().cast(),
            }
        }
    };
}
