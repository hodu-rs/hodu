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

/// Plugin entry point function type (returns raw pointer for FFI safety)
/// Each plugin must export: `extern "C" fn hodu_format_plugin_create() -> *mut ()`
/// The returned pointer should be cast to `Box<dyn FormatPlugin>` using `from_raw`
pub type FormatPluginCreateFn = unsafe extern "C" fn() -> *mut ();

/// Plugin destroy function type
/// Each plugin must export: `extern "C" fn hodu_format_plugin_destroy(ptr: *mut ())`
pub type FormatPluginDestroyFn = unsafe extern "C" fn(ptr: *mut ());
