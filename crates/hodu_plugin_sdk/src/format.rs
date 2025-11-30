//! Format plugin interface for file format support
//!
//! A `FormatPlugin` handles loading and saving models (Snapshot) and tensors (TensorData)
//! in various file formats like ONNX, safetensors, npy, etc.

use crate::{HoduResult, Snapshot, TensorData};
use std::path::Path;

/// Capabilities flags for format plugins
#[derive(Debug, Clone, Copy, Default)]
pub struct FormatCapabilities {
    /// Can load models from this format
    pub load_model: bool,
    /// Can save models to this format
    pub save_model: bool,
    /// Can load tensors from this format
    pub load_tensor: bool,
    /// Can save tensors to this format
    pub save_tensor: bool,
}

impl FormatCapabilities {
    /// Create full capabilities (all operations supported)
    pub fn full() -> Self {
        Self {
            load_model: true,
            save_model: true,
            load_tensor: true,
            save_tensor: true,
        }
    }

    /// Create model-only capabilities
    pub fn model_only() -> Self {
        Self {
            load_model: true,
            save_model: true,
            load_tensor: false,
            save_tensor: false,
        }
    }

    /// Create tensor-only capabilities
    pub fn tensor_only() -> Self {
        Self {
            load_model: false,
            save_model: false,
            load_tensor: true,
            save_tensor: true,
        }
    }

    /// Create load-only capabilities (read-only format)
    pub fn load_only() -> Self {
        Self {
            load_model: true,
            save_model: false,
            load_tensor: true,
            save_tensor: false,
        }
    }
}

/// Format plugin interface
///
/// Handles file format conversion for models and tensors.
/// The CLI automatically selects plugins based on file extension.
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
///
///     fn capabilities(&self) -> FormatCapabilities {
///         FormatCapabilities::tensor_only()
///     }
///
///     fn supported_extensions(&self) -> Vec<&str> {
///         vec!["npy", "npz"]
///     }
///
///     // Model operations - not supported
///     fn load_model(&self, _: &Path) -> HoduResult<Snapshot> {
///         Err(HoduError::BackendError("npy does not support models".into()))
///     }
///     // ... other model methods return errors
///
///     // Tensor operations - supported
///     fn load_tensor(&self, path: &Path) -> HoduResult<TensorData> {
///         // ... implementation
///     }
///     // ... other tensor methods
/// }
/// ```
pub trait FormatPlugin: Send + Sync {
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
    fn load_model(&self, path: &Path) -> HoduResult<Snapshot>;

    /// Load a model from bytes
    fn load_model_from_bytes(&self, data: &[u8]) -> HoduResult<Snapshot>;

    /// Save a model to file
    fn save_model(&self, snapshot: &Snapshot, path: &Path) -> HoduResult<()>;

    /// Save a model to bytes
    fn save_model_to_bytes(&self, snapshot: &Snapshot) -> HoduResult<Vec<u8>>;

    // === Tensor operations ===

    /// Load a tensor from file
    fn load_tensor(&self, path: &Path) -> HoduResult<TensorData>;

    /// Load a tensor from bytes
    fn load_tensor_from_bytes(&self, data: &[u8]) -> HoduResult<TensorData>;

    /// Save a tensor to file
    fn save_tensor(&self, tensor: &TensorData, path: &Path) -> HoduResult<()>;

    /// Save a tensor to bytes
    fn save_tensor_to_bytes(&self, tensor: &TensorData) -> HoduResult<Vec<u8>>;

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
        self.capabilities().load_model
    }

    /// Check if this plugin can save models
    fn can_save_model(&self) -> bool {
        self.capabilities().save_model
    }

    /// Check if this plugin can load tensors
    fn can_load_tensor(&self) -> bool {
        self.capabilities().load_tensor
    }

    /// Check if this plugin can save tensors
    fn can_save_tensor(&self) -> bool {
        self.capabilities().save_tensor
    }
}

/// Plugin metadata for SDK version verification
#[repr(C)]
pub struct FormatPluginMetadata {
    /// SDK version used to compile this plugin (semver)
    pub sdk_version: *const std::ffi::c_char,
    /// Plugin name
    pub name: *const std::ffi::c_char,
    /// Plugin version
    pub version: *const std::ffi::c_char,
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

/// Macro to export format plugin FFI functions
///
/// # Example
///
/// ```ignore
/// use hodu_plugin_sdk::*;
///
/// #[derive(Default)]
/// pub struct OnnxFormat;
/// impl FormatPlugin for OnnxFormat { /* ... */ }
///
/// export_format_plugin!(OnnxFormat, "onnx", "0.1.0");
/// ```
#[macro_export]
macro_rules! export_format_plugin {
    ($plugin_type:ty, $name:literal, $version:literal) => {
        #[no_mangle]
        pub extern "C" fn hodu_format_plugin_create() -> *mut $crate::FormatPluginHandle {
            let boxed: Box<dyn $crate::FormatPlugin> = Box::new(<$plugin_type>::default());
            $crate::FormatPluginHandle::from_boxed(boxed)
        }

        #[no_mangle]
        pub unsafe extern "C" fn hodu_format_plugin_destroy(ptr: *mut $crate::FormatPluginHandle) {
            if !ptr.is_null() {
                drop($crate::FormatPluginHandle::into_boxed(ptr));
            }
        }

        #[no_mangle]
        pub extern "C" fn hodu_format_plugin_metadata() -> $crate::FormatPluginMetadata {
            $crate::FormatPluginMetadata {
                sdk_version: concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast(),
                name: concat!($name, "\0").as_ptr().cast(),
                version: concat!($version, "\0").as_ptr().cast(),
            }
        }
    };
}
