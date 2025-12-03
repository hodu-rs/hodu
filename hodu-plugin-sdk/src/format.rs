//! Format plugin interface for file format support
//!
//! A `FormatPlugin` handles loading and saving models (Snapshot) and tensors (TensorData)
//! in various file formats like ONNX, safetensors, npy, etc.

use crate::{PluginResult, Snapshot, TensorData};
use std::path::Path;

/// Capabilities flags for format plugins
///
/// This uses a bitflags pattern for forward compatibility. New capability flags
/// can be added in future SDK versions without breaking existing plugins.
///
/// # Example
/// ```ignore
/// let caps = FormatCapabilities::LOAD_MODEL | FormatCapabilities::SAVE_MODEL;
/// assert!(caps.has_load_model());
/// assert!(caps.has_save_model());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct FormatCapabilities(u32);

impl FormatCapabilities {
    /// No capabilities
    pub const NONE: Self = Self(0);
    /// Can load models from this format
    pub const LOAD_MODEL: Self = Self(1 << 0);
    /// Can save models to this format
    pub const SAVE_MODEL: Self = Self(1 << 1);
    /// Can load tensors from this format
    pub const LOAD_TENSOR: Self = Self(1 << 2);
    /// Can save tensors to this format
    pub const SAVE_TENSOR: Self = Self(1 << 3);
    // Reserved for future capabilities:
    // pub const LOAD_WEIGHTS: Self = Self(1 << 4);
    // pub const SAVE_WEIGHTS: Self = Self(1 << 5);

    /// Create full capabilities (all operations supported)
    pub const fn full() -> Self {
        Self(Self::LOAD_MODEL.0 | Self::SAVE_MODEL.0 | Self::LOAD_TENSOR.0 | Self::SAVE_TENSOR.0)
    }

    /// Create model-only capabilities
    pub const fn model_only() -> Self {
        Self(Self::LOAD_MODEL.0 | Self::SAVE_MODEL.0)
    }

    /// Create tensor-only capabilities
    pub const fn tensor_only() -> Self {
        Self(Self::LOAD_TENSOR.0 | Self::SAVE_TENSOR.0)
    }

    /// Create load-only capabilities (read-only format)
    pub const fn load_only() -> Self {
        Self(Self::LOAD_MODEL.0 | Self::LOAD_TENSOR.0)
    }

    /// Create from raw bits (for FFI)
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Get raw bits (for FFI)
    pub const fn bits(&self) -> u32 {
        self.0
    }

    /// Check if load_model capability is set
    pub const fn has_load_model(&self) -> bool {
        self.0 & Self::LOAD_MODEL.0 != 0
    }

    /// Check if save_model capability is set
    pub const fn has_save_model(&self) -> bool {
        self.0 & Self::SAVE_MODEL.0 != 0
    }

    /// Check if load_tensor capability is set
    pub const fn has_load_tensor(&self) -> bool {
        self.0 & Self::LOAD_TENSOR.0 != 0
    }

    /// Check if save_tensor capability is set
    pub const fn has_save_tensor(&self) -> bool {
        self.0 & Self::SAVE_TENSOR.0 != 0
    }

    // Backwards compatibility properties
    /// Can load models from this format
    #[deprecated(since = "0.4.0", note = "use has_load_model() instead")]
    pub const fn load_model(&self) -> bool {
        self.has_load_model()
    }

    /// Can save models to this format
    #[deprecated(since = "0.4.0", note = "use has_save_model() instead")]
    pub const fn save_model(&self) -> bool {
        self.has_save_model()
    }

    /// Can load tensors from this format
    #[deprecated(since = "0.4.0", note = "use has_load_tensor() instead")]
    pub const fn load_tensor(&self) -> bool {
        self.has_load_tensor()
    }

    /// Can save tensors to this format
    #[deprecated(since = "0.4.0", note = "use has_save_tensor() instead")]
    pub const fn save_tensor(&self) -> bool {
        self.has_save_tensor()
    }
}

impl std::ops::BitOr for FormatCapabilities {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitAnd for FormatCapabilities {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

/// Format plugin interface
///
/// Handles file format conversion for models and tensors.
/// The CLI automatically selects plugins based on file extension.
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
/// use hodu_plugin_sdk::*;
///
/// pub struct NpyFormat;
///
/// impl FormatPlugin for NpyFormat {
///     fn name(&self) -> &str { "npy" }
///     fn version(&self) -> &str { "0.1.0" }
///     fn capabilities(&self) -> FormatCapabilities {
///         FormatCapabilities::tensor_only()
///     }
///     fn supported_extensions(&self) -> Vec<&str> {
///         vec!["npy", "npz"]
///     }
///     // Only implement tensor operations - model operations use defaults
///     fn load_tensor(&self, path: &Path) -> PluginResult<TensorData> {
///         // ... implementation
///     }
///     fn load_tensor_from_bytes(&self, data: &[u8]) -> PluginResult<TensorData> {
///         // ... implementation
///     }
///     // ... other tensor methods
/// }
/// ```
pub trait FormatPlugin: Send + Sync {
    // === Required methods ===

    /// Plugin name (e.g., "onnx", "safetensors", "npy")
    fn name(&self) -> &str;

    /// Plugin version (semver)
    fn version(&self) -> &str;

    /// Capabilities provided by this plugin
    fn capabilities(&self) -> FormatCapabilities;

    /// File extensions supported by this plugin (without dot)
    fn supported_extensions(&self) -> Vec<&str>;

    // === Model operations ===

    /// Load a model from file
    ///
    /// Default: returns `NotSupported` error
    #[allow(unused_variables)]
    fn load_model(&self, path: &Path) -> PluginResult<Snapshot> {
        Err(crate::PluginError::NotSupported("load_model".into()))
    }

    /// Load a model from bytes
    ///
    /// Default: returns `NotSupported` error
    #[allow(unused_variables)]
    fn load_model_from_bytes(&self, data: &[u8]) -> PluginResult<Snapshot> {
        Err(crate::PluginError::NotSupported("load_model_from_bytes".into()))
    }

    /// Save a model to file
    ///
    /// Default: returns `NotSupported` error
    #[allow(unused_variables)]
    fn save_model(&self, snapshot: &Snapshot, path: &Path) -> PluginResult<()> {
        Err(crate::PluginError::NotSupported("save_model".into()))
    }

    /// Save a model to bytes
    ///
    /// Default: returns `NotSupported` error
    #[allow(unused_variables)]
    fn save_model_to_bytes(&self, snapshot: &Snapshot) -> PluginResult<Vec<u8>> {
        Err(crate::PluginError::NotSupported("save_model_to_bytes".into()))
    }

    // === Tensor operations ===

    /// Load a tensor from file
    ///
    /// Default: returns `NotSupported` error
    #[allow(unused_variables)]
    fn load_tensor(&self, path: &Path) -> PluginResult<TensorData> {
        Err(crate::PluginError::NotSupported("load_tensor".into()))
    }

    /// Load a tensor from bytes
    ///
    /// Default: returns `NotSupported` error
    #[allow(unused_variables)]
    fn load_tensor_from_bytes(&self, data: &[u8]) -> PluginResult<TensorData> {
        Err(crate::PluginError::NotSupported("load_tensor_from_bytes".into()))
    }

    /// Save a tensor to file
    ///
    /// Default: returns `NotSupported` error
    #[allow(unused_variables)]
    fn save_tensor(&self, tensor: &TensorData, path: &Path) -> PluginResult<()> {
        Err(crate::PluginError::NotSupported("save_tensor".into()))
    }

    /// Save a tensor to bytes
    ///
    /// Default: returns `NotSupported` error
    #[allow(unused_variables)]
    fn save_tensor_to_bytes(&self, tensor: &TensorData) -> PluginResult<Vec<u8>> {
        Err(crate::PluginError::NotSupported("save_tensor_to_bytes".into()))
    }

    // === Convenience methods ===

    /// Check if this plugin supports a file extension
    fn supports_extension(&self, ext: &str) -> bool {
        let ext_lower = ext.to_lowercase();
        self.supported_extensions()
            .iter()
            .any(|e| e.eq_ignore_ascii_case(&ext_lower))
    }

    /// Check if this plugin can load models
    fn can_load_model(&self) -> bool {
        self.capabilities().has_load_model()
    }

    /// Check if this plugin can save models
    fn can_save_model(&self) -> bool {
        self.capabilities().has_save_model()
    }

    /// Check if this plugin can load tensors
    fn can_load_tensor(&self) -> bool {
        self.capabilities().has_load_tensor()
    }

    /// Check if this plugin can save tensors
    fn can_save_tensor(&self) -> bool {
        self.capabilities().has_save_tensor()
    }
}

/// Plugin metadata for SDK version verification
///
/// This struct is `#[repr(C)]` for FFI stability. Reserved fields ensure
/// forward compatibility when new metadata fields are added.
#[repr(C)]
pub struct FormatPluginMetadata {
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
/// This wraps a `Box<dyn FormatPlugin>` but is represented as a raw pointer
/// for FFI safety. The actual type is erased at the C boundary.
#[repr(C)]
pub struct FormatPluginHandle {
    _opaque: [u8; 0],
}

/// Plugin entry point function type
/// Each plugin must export: `extern "C" fn hodu_format_plugin_create() -> *mut FormatPluginHandle`
pub type FormatPluginCreateFn = unsafe extern "C" fn() -> *mut FormatPluginHandle;

/// Plugin destroy function type
/// Each plugin must export: `extern "C" fn hodu_format_plugin_destroy(ptr: *mut FormatPluginHandle)`
pub type FormatPluginDestroyFn = unsafe extern "C" fn(ptr: *mut FormatPluginHandle);

/// Plugin metadata function type
/// Each plugin must export: `extern "C" fn hodu_format_plugin_metadata() -> FormatPluginMetadata`
pub type FormatPluginMetadataFn = unsafe extern "C" fn() -> FormatPluginMetadata;

impl FormatPluginHandle {
    /// Create a handle from a boxed plugin (called from plugin side)
    pub fn from_boxed(plugin: Box<dyn FormatPlugin>) -> *mut Self {
        Box::into_raw(Box::new(plugin)) as *mut Self
    }

    /// Convert handle back to boxed plugin (called from host side)
    ///
    /// # Safety
    /// The handle must have been created by `from_boxed` and not yet destroyed
    pub unsafe fn into_boxed(ptr: *mut Self) -> Box<Box<dyn FormatPlugin>> {
        Box::from_raw(ptr as *mut Box<dyn FormatPlugin>)
    }

    /// Get a reference to the plugin
    ///
    /// # Safety
    /// The handle must be valid
    pub unsafe fn as_ref<'a>(ptr: *mut Self) -> &'a dyn FormatPlugin {
        let boxed = &*(ptr as *mut Box<dyn FormatPlugin>);
        boxed.as_ref()
    }
}
