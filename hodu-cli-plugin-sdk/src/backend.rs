//! Backend plugin interface for model execution and AOT compilation
//!
//! A `BackendPlugin` combines Runner (execution) and Builder (AOT compilation) capabilities.
//! Plugins can implement one or both functionalities based on their `BackendCapabilities`.

use crate::{BuildFormat, PluginResult, Snapshot, TensorData};
use hodu_compat::*;
use std::path::Path;

/// Target device for plugin execution
///
/// This represents which backend/hardware a plugin targets.
/// It's independent from `hodu_core::Device` (which represents tensor memory location).
///
/// Note: This enum is `#[non_exhaustive]` - new devices may be added in future versions.
/// Always include a wildcard pattern when matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Device {
    /// CPU execution
    CPU,
    /// CUDA GPU execution with device ID
    CUDA(usize),
    /// Metal GPU execution (Apple Silicon)
    Metal,
}

impl Device {
    /// Parse device from string (e.g., "CPU", "CUDA:0", "Metal")
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "CPU" => Some(Device::CPU),
            "Metal" => Some(Device::Metal),
            s if s.starts_with("CUDA") => {
                let id = s.strip_prefix("CUDA:").and_then(|id| id.parse().ok()).unwrap_or(0);
                Some(Device::CUDA(id))
            },
            _ => None,
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::CPU => write!(f, "CPU"),
            Device::CUDA(id) => write!(f, "CUDA:{}", id),
            Device::Metal => write!(f, "Metal"),
        }
    }
}

/// Capabilities flags for backend plugins
///
/// This uses a bitflags pattern for forward compatibility. New capability flags
/// can be added in future SDK versions without breaking existing plugins.
///
/// # Example
/// ```ignore
/// let caps = BackendCapabilities::RUNNER | BackendCapabilities::BUILDER;
/// assert!(caps.has_runner());
/// assert!(caps.has_builder());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct BackendCapabilities(u32);

impl BackendCapabilities {
    /// No capabilities
    pub const NONE: Self = Self(0);
    /// Supports running models (`hodu run`)
    pub const RUNNER: Self = Self(1 << 0);
    /// Supports building artifacts (`hodu build`)
    pub const BUILDER: Self = Self(1 << 1);
    // Reserved for future capabilities:
    // pub const PROFILER: Self = Self(1 << 2);
    // pub const DEBUGGER: Self = Self(1 << 3);

    /// Create capabilities with both runner and builder
    pub const fn full() -> Self {
        Self(Self::RUNNER.0 | Self::BUILDER.0)
    }

    /// Create runner-only capabilities
    pub const fn runner_only() -> Self {
        Self::RUNNER
    }

    /// Create builder-only capabilities
    pub const fn builder_only() -> Self {
        Self::BUILDER
    }

    /// Create from raw bits (for FFI)
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Get raw bits (for FFI)
    pub const fn bits(&self) -> u32 {
        self.0
    }

    /// Check if runner capability is set
    pub const fn has_runner(&self) -> bool {
        self.0 & Self::RUNNER.0 != 0
    }

    /// Check if builder capability is set
    pub const fn has_builder(&self) -> bool {
        self.0 & Self::BUILDER.0 != 0
    }

    // Backwards compatibility properties
    /// Whether this plugin supports running models (`hodu run`)
    #[deprecated(since = "0.4.0", note = "use has_runner() instead")]
    pub const fn runner(&self) -> bool {
        self.has_runner()
    }

    /// Whether this plugin supports building artifacts (`hodu build`)
    #[deprecated(since = "0.4.0", note = "use has_builder() instead")]
    pub const fn builder(&self) -> bool {
        self.has_builder()
    }
}

impl std::ops::BitOr for BackendCapabilities {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitAnd for BackendCapabilities {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
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
/// # Versioning
///
/// This trait provides default implementations for optional methods to ensure
/// forward compatibility. When new methods are added in future SDK versions,
/// existing plugins will continue to work with the default behavior.
///
/// # Examples
///
/// ```ignore
/// use hodu_cli_plugin_sdk::*;
///
/// pub struct MyBackend;
///
/// impl BackendPlugin for MyBackend {
///     fn name(&self) -> &str { "my-backend" }
///     fn version(&self) -> &str { "0.1.0" }
///     fn capabilities(&self) -> BackendCapabilities {
///         BackendCapabilities::runner_only()
///     }
///     fn supported_devices(&self) -> Vec<Device> {
///         vec![Device::CPU]
///     }
///     fn run(&self, snapshot: &Snapshot, device: Device, inputs: &[(&str, TensorData)])
///         -> PluginResult<HashMap<String, TensorData>> {
///         // ... implementation
///     }
///     // Builder methods use defaults (return NotSupported)
/// }
/// ```
pub trait BackendPlugin: Send + Sync {
    // === Required methods ===

    /// Plugin name (e.g., "cpu", "metal", "cuda", "llvm")
    fn name(&self) -> &str;

    /// Plugin version (semver)
    fn version(&self) -> &str;

    /// Capabilities provided by this plugin
    fn capabilities(&self) -> BackendCapabilities;

    // === Runner functionality (`hodu run`) ===

    /// Devices supported for running models
    ///
    /// Default: empty (no devices supported)
    fn supported_devices(&self) -> Vec<Device> {
        vec![]
    }

    /// Run a model on the specified device
    ///
    /// # Arguments
    /// * `snapshot` - The model to execute
    /// * `device` - Target device (must be in `supported_devices()`)
    /// * `inputs` - Named input tensors
    ///
    /// # Returns
    /// Named output tensors
    ///
    /// Default: returns `NotSupported` error
    #[allow(unused_variables)]
    fn run(
        &self,
        snapshot: &Snapshot,
        device: Device,
        inputs: &[(&str, TensorData)],
    ) -> PluginResult<HashMap<String, TensorData>> {
        Err(crate::PluginError::NotSupported("run".into()))
    }

    // === Builder functionality (`hodu build`) ===

    /// Build targets supported for AOT compilation
    ///
    /// Default: empty (no targets supported)
    fn supported_targets(&self) -> Vec<BuildTarget> {
        vec![]
    }

    /// Output formats supported for a given build target
    ///
    /// Default: empty (no formats supported)
    #[allow(unused_variables)]
    fn supported_formats(&self, target: &BuildTarget) -> Vec<BuildFormat> {
        vec![]
    }

    /// Build an AOT artifact
    ///
    /// # Arguments
    /// * `snapshot` - The model to compile
    /// * `target` - Target specification (triple + device)
    /// * `format` - Output format (must be in `supported_formats(target)`)
    /// * `output` - Output file path
    ///
    /// Default: returns `NotSupported` error
    #[allow(unused_variables)]
    fn build(&self, snapshot: &Snapshot, target: &BuildTarget, format: BuildFormat, output: &Path) -> PluginResult<()> {
        Err(crate::PluginError::NotSupported("build".into()))
    }

    // === Convenience methods ===

    /// Check if this plugin supports a device for running
    fn supports_device(&self, device: Device) -> bool {
        self.capabilities().has_runner() && self.supported_devices().contains(&device)
    }

    /// Check if this plugin supports a build target
    fn supports_target(&self, target: &BuildTarget) -> bool {
        self.capabilities().has_builder() && self.supported_targets().contains(target)
    }
}

/// Plugin metadata for SDK version verification
///
/// This struct is `#[repr(C)]` for FFI stability. Reserved fields ensure
/// forward compatibility when new metadata fields are added.
#[repr(C)]
pub struct BackendPluginMetadata {
    /// FFI protocol version for ABI compatibility
    pub ffi_version: u32,
    /// SDK version used to compile this plugin (semver, null-terminated)
    pub sdk_version: *const std::ffi::c_char,
    /// Plugin name (null-terminated)
    pub name: *const std::ffi::c_char,
    /// Plugin version (null-terminated)
    pub version: *const std::ffi::c_char,
    /// Reserved for future use (must be zeroed)
    pub _reserved: [usize; 4],
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
