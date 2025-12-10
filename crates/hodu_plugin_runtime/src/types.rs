//! Plugin types and data structures

use serde::{Deserialize, Serialize};

/// A single plugin entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginEntry {
    /// Plugin name (e.g., "hodu-backend-interp")
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Plugin license (e.g., "MIT", "Apache-2.0")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Plugin type
    #[serde(rename = "type")]
    pub plugin_type: PluginType,
    /// Plugin capabilities
    pub capabilities: PluginCapabilities,
    /// Executable binary filename (e.g., "hodu-plugin-onnx")
    pub binary: String,
    /// Installation source
    pub source: PluginSource,
    /// Installation timestamp (ISO 8601)
    pub installed_at: String,
    /// Plugin protocol version used to build
    pub plugin_version: String,
    /// Whether the plugin is enabled (default: true)
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Plugin dependencies (other plugin names)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<String>,
}

fn default_enabled() -> bool {
    true
}

/// Plugin type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginType {
    Backend,
    ModelFormat,
    TensorFormat,
}

/// Plugin capabilities (union of backend and format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginCapabilities {
    // Backend capabilities
    /// Supports `backend.run` RPC method
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runner: Option<bool>,
    /// Supports `backend.build` RPC method
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub builder: Option<bool>,
    /// Supported devices (e.g., "cpu", "cuda", "metal")
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub devices: Vec<String>,
    /// Supported build targets
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<String>,

    // Format capabilities
    /// Supports `format.load_model` RPC method
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_model: Option<bool>,
    /// Supports `format.save_model` RPC method
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_model: Option<bool>,
    /// Supports `format.load_tensor` RPC method
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_tensor: Option<bool>,
    /// Supports `format.save_tensor` RPC method
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_tensor: Option<bool>,
    /// Supported model file extensions (e.g., ["onnx", "pb"])
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub model_extensions: Vec<String>,
    /// Supported tensor file extensions (e.g., ["npy", "npz"])
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tensor_extensions: Vec<String>,
}

impl PluginCapabilities {
    /// Create backend capabilities
    pub fn backend(runner: bool, builder: bool, devices: Vec<String>, targets: Vec<String>) -> Self {
        Self {
            runner: Some(runner),
            builder: Some(builder),
            devices,
            targets,
            load_model: None,
            save_model: None,
            load_tensor: None,
            save_tensor: None,
            model_extensions: Vec::new(),
            tensor_extensions: Vec::new(),
        }
    }

    /// Create model format capabilities
    pub fn model_format(load_model: bool, save_model: bool, model_extensions: Vec<String>) -> Self {
        Self {
            runner: None,
            builder: None,
            devices: Vec::new(),
            targets: Vec::new(),
            load_model: Some(load_model),
            save_model: Some(save_model),
            load_tensor: None,
            save_tensor: None,
            model_extensions,
            tensor_extensions: Vec::new(),
        }
    }

    /// Create tensor format capabilities
    pub fn tensor_format(load_tensor: bool, save_tensor: bool, tensor_extensions: Vec<String>) -> Self {
        Self {
            runner: None,
            builder: None,
            devices: Vec::new(),
            targets: Vec::new(),
            load_model: None,
            save_model: None,
            load_tensor: Some(load_tensor),
            save_tensor: Some(save_tensor),
            model_extensions: Vec::new(),
            tensor_extensions,
        }
    }
}

/// Installation source
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum PluginSource {
    /// From crates.io
    CratesIo,
    /// From git repository
    Git {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        tag: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        subdir: Option<String>,
    },
    /// From local path
    Local { path: String },
}

impl std::fmt::Display for PluginSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginSource::CratesIo => write!(f, "crates.io"),
            PluginSource::Git { url, tag, .. } => {
                if let Some(t) = tag {
                    write!(f, "git:{}@{}", url, t)
                } else {
                    write!(f, "git:{}", url)
                }
            },
            PluginSource::Local { path } => write!(f, "local:{}", path),
        }
    }
}

/// Detected plugin type with metadata (from manifest or initialization)
#[derive(Debug, Clone)]
pub enum DetectedPluginType {
    Backend {
        name: String,
        version: String,
        plugin_version: String,
    },
    ModelFormat {
        name: String,
        version: String,
        plugin_version: String,
    },
    TensorFormat {
        name: String,
        version: String,
        plugin_version: String,
    },
}
