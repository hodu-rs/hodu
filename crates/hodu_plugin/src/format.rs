//! Format plugin interface for model loading and saving

use crate::{HoduResult, Script};
use std::path::Path;

/// Format plugin interface
///
/// A format plugin handles loading and saving models in specific formats.
/// Examples: ONNX, SafeTensors, GGUF, PyTorch
pub trait FormatPlugin: Send + Sync {
    /// Plugin name (e.g., "onnx", "safetensors", "gguf")
    fn name(&self) -> &str;

    /// Plugin version
    fn version(&self) -> &str;

    /// Supported file extensions (e.g., ["onnx"], ["safetensors"], ["gguf"])
    fn extensions(&self) -> &[&str];

    /// Check if this plugin can handle the given extension
    fn supports_extension(&self, ext: &str) -> bool {
        self.extensions().iter().any(|e| e.eq_ignore_ascii_case(ext))
    }

    /// Load a model from file and convert to Script
    fn load(&self, path: &Path) -> HoduResult<Script>;

    /// Save a Script to file (optional, not all formats support export)
    fn save(&self, script: &Script, path: &Path) -> HoduResult<()> {
        let _ = (script, path);
        Err(hodu_core::error::HoduError::UnsupportedOperation(
            format!("Format '{}' does not support saving", self.name()).into(),
        ))
    }

    /// Check if this format supports saving
    fn can_save(&self) -> bool {
        false
    }
}

/// Plugin information for discovery
#[repr(C)]
pub struct FormatPluginInfo {
    pub name: &'static str,
    pub version: &'static str,
    pub extensions: &'static [&'static str],
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
