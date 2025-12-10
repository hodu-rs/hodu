//! Plugin system for Hodu CLI
//!
//! This module re-exports the plugin runtime and provides CLI-specific
//! notification handling.

// Re-export from hodu_plugin_runtime
pub use hodu_plugin_runtime::backend;
pub use hodu_plugin_runtime::format;
pub use hodu_plugin_runtime::{
    detect_plugin_type, CancellationHandle, ClientError, DetectedPluginType, PluginCapabilities, PluginClient,
    PluginDetectError, PluginEntry, PluginRegistry, PluginSource, PluginType, RegistryError, DEFAULT_TIMEOUT,
};

mod process;

pub use process::*;
