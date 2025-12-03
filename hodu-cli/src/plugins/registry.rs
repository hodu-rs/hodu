use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Plugin registry stored in ~/.hodu/plugins.json
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginRegistry {
    /// Schema version for future compatibility
    pub version: u32,
    /// Installed plugins
    pub plugins: Vec<PluginEntry>,
}

/// A single plugin entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginEntry {
    /// Plugin name (e.g., "hodu-backend-interp")
    pub name: String,
    /// Plugin version
    pub version: String,
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
    /// SDK version used to build
    pub sdk_version: String,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runner: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub builder: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub devices: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<String>,

    // Format capabilities
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_model: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_model: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_tensor: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_tensor: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub model_extensions: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tensor_extensions: Vec<String>,
}

/// Installation source
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
#[serde(rename_all = "lowercase")]
pub enum PluginSource {
    /// From crates.io
    CratesIo,
    /// From git repository
    Git(String),
    /// From local path
    Local(String),
}

impl PluginRegistry {
    /// Current schema version
    pub const CURRENT_VERSION: u32 = 1;

    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            plugins: Vec::new(),
        }
    }

    /// Load registry from file
    pub fn load(path: &Path) -> Result<Self, RegistryError> {
        if !path.exists() {
            return Ok(Self::new());
        }

        let content = std::fs::read_to_string(path).map_err(|e| RegistryError::Io(e.to_string()))?;

        let registry: Self = serde_json::from_str(&content).map_err(|e| RegistryError::Parse(e.to_string()))?;

        Ok(registry)
    }

    /// Save registry to file
    pub fn save(&self, path: &Path) -> Result<(), RegistryError> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| RegistryError::Io(e.to_string()))?;
        }

        let content = serde_json::to_string_pretty(self).map_err(|e| RegistryError::Serialize(e.to_string()))?;

        std::fs::write(path, content).map_err(|e| RegistryError::Io(e.to_string()))?;

        Ok(())
    }

    /// Get default registry path (~/.hodu/plugins.json)
    pub fn default_path() -> Result<PathBuf, RegistryError> {
        let home = dirs::home_dir().ok_or(RegistryError::NoHomeDir)?;
        Ok(home.join(".hodu").join("plugins.json"))
    }

    /// Get plugins directory (~/.hodu/plugins/)
    pub fn plugins_dir() -> Result<PathBuf, RegistryError> {
        let home = dirs::home_dir().ok_or(RegistryError::NoHomeDir)?;
        Ok(home.join(".hodu").join("plugins"))
    }

    /// Find plugin by name
    pub fn find(&self, name: &str) -> Option<&PluginEntry> {
        self.plugins.iter().find(|p| p.name == name)
    }

    /// Find plugin by name (mutable)
    pub fn find_mut(&mut self, name: &str) -> Option<&mut PluginEntry> {
        self.plugins.iter_mut().find(|p| p.name == name)
    }

    /// Add or update a plugin
    pub fn upsert(&mut self, entry: PluginEntry) {
        if let Some(existing) = self.find_mut(&entry.name) {
            *existing = entry;
        } else {
            self.plugins.push(entry);
        }
    }

    /// Remove a plugin by name
    pub fn remove(&mut self, name: &str) -> Option<PluginEntry> {
        if let Some(idx) = self.plugins.iter().position(|p| p.name == name) {
            Some(self.plugins.remove(idx))
        } else {
            None
        }
    }

    /// List backend plugins
    pub fn backends(&self) -> impl Iterator<Item = &PluginEntry> {
        self.plugins.iter().filter(|p| p.plugin_type == PluginType::Backend)
    }

    /// List model format plugins
    pub fn model_formats(&self) -> impl Iterator<Item = &PluginEntry> {
        self.plugins.iter().filter(|p| p.plugin_type == PluginType::ModelFormat)
    }

    /// List tensor format plugins
    pub fn tensor_formats(&self) -> impl Iterator<Item = &PluginEntry> {
        self.plugins
            .iter()
            .filter(|p| p.plugin_type == PluginType::TensorFormat)
    }

    /// Find backend plugin by device
    /// Device comparison is case-insensitive (normalized to lowercase)
    pub fn find_backend_by_device(&self, device: &str) -> Option<&PluginEntry> {
        let device_lower = device.to_lowercase();
        self.backends()
            .find(|p| p.capabilities.devices.iter().any(|d| d.to_lowercase() == device_lower))
    }

    /// Find model format plugin by extension
    /// Extension comparison normalizes to lowercase without leading dot
    pub fn find_model_format_by_extension(&self, ext: &str) -> Option<&PluginEntry> {
        let ext_normalized = ext.trim_start_matches('.').to_lowercase();
        self.model_formats().find(|p| {
            p.capabilities
                .model_extensions
                .iter()
                .any(|e| e.trim_start_matches('.').to_lowercase() == ext_normalized)
        })
    }

    /// Find tensor format plugin by extension
    /// Extension comparison normalizes to lowercase without leading dot
    pub fn find_tensor_format_by_extension(&self, ext: &str) -> Option<&PluginEntry> {
        let ext_normalized = ext.trim_start_matches('.').to_lowercase();
        self.tensor_formats().find(|p| {
            p.capabilities
                .tensor_extensions
                .iter()
                .any(|e| e.trim_start_matches('.').to_lowercase() == ext_normalized)
        })
    }
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

impl std::fmt::Display for PluginSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginSource::CratesIo => write!(f, "crates.io"),
            PluginSource::Git(url) => write!(f, "git:{}", url),
            PluginSource::Local(path) => write!(f, "local:{}", path),
        }
    }
}

/// Registry errors
#[derive(Debug)]
pub enum RegistryError {
    NoHomeDir,
    Io(String),
    Parse(String),
    Serialize(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::NoHomeDir => write!(f, "Could not find home directory"),
            RegistryError::Io(e) => write!(f, "IO error: {}", e),
            RegistryError::Parse(e) => write!(f, "Parse error: {}", e),
            RegistryError::Serialize(e) => write!(f, "Serialize error: {}", e),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Detected plugin type with metadata (from manifest or initialization)
#[derive(Debug, Clone)]
pub enum DetectedPluginType {
    Backend {
        name: String,
        version: String,
        sdk_version: String,
    },
    ModelFormat {
        name: String,
        version: String,
        sdk_version: String,
    },
    TensorFormat {
        name: String,
        version: String,
        sdk_version: String,
    },
}

/// Plugin detection errors
#[derive(Debug)]
pub enum PluginDetectError {
    NotAPlugin,
    IoError(String),
    ParseError(String),
}

impl std::fmt::Display for PluginDetectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginDetectError::NotAPlugin => write!(f, "Not a valid plugin"),
            PluginDetectError::IoError(e) => write!(f, "IO error: {}", e),
            PluginDetectError::ParseError(e) => write!(f, "Parse error: {}", e),
        }
    }
}

impl std::error::Error for PluginDetectError {}

/// Detect plugin type by reading manifest.json or spawning the plugin
///
/// For JSON-RPC based plugins, this spawns the plugin executable,
/// sends an initialize request, and reads the capabilities from the response.
pub fn detect_plugin_type(path: &std::path::Path) -> Result<DetectedPluginType, PluginDetectError> {
    // First try to read manifest.json from the plugin directory
    let manifest_path = path
        .parent()
        .map(|p| p.join("manifest.json"))
        .unwrap_or_else(|| path.with_file_name("manifest.json"));

    if manifest_path.exists() {
        let content = std::fs::read_to_string(&manifest_path).map_err(|e| PluginDetectError::IoError(e.to_string()))?;
        let manifest: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| PluginDetectError::ParseError(e.to_string()))?;

        let name = manifest["name"].as_str().unwrap_or("unknown").to_string();
        let version = manifest["version"].as_str().unwrap_or("0.0.0").to_string();
        let sdk_version = manifest["sdk_version"].as_str().unwrap_or("0.1.0").to_string();
        let capabilities = manifest["capabilities"].as_array();

        // Determine type from capabilities
        let empty_caps = vec![];
        let caps = capabilities.unwrap_or(&empty_caps);
        let has_backend = caps
            .iter()
            .any(|c| c.as_str().map(|s| s.starts_with("backend.")).unwrap_or(false));
        let has_model_format = caps.iter().any(|c| {
            c.as_str()
                .map(|s| s == "format.load_model" || s == "format.save_model")
                .unwrap_or(false)
        });
        let has_tensor_format = caps.iter().any(|c| {
            c.as_str()
                .map(|s| s == "format.load_tensor" || s == "format.save_tensor")
                .unwrap_or(false)
        });

        if has_backend {
            return Ok(DetectedPluginType::Backend {
                name,
                version,
                sdk_version,
            });
        } else if has_model_format {
            return Ok(DetectedPluginType::ModelFormat {
                name,
                version,
                sdk_version,
            });
        } else if has_tensor_format {
            return Ok(DetectedPluginType::TensorFormat {
                name,
                version,
                sdk_version,
            });
        } else {
            // Default to ModelFormat if type cannot be determined
            return Ok(DetectedPluginType::ModelFormat {
                name,
                version,
                sdk_version,
            });
        }
    }

    // If no manifest, try to spawn the plugin and initialize it
    // This is a simplified stub - full implementation would spawn the process
    Err(PluginDetectError::NotAPlugin)
}
