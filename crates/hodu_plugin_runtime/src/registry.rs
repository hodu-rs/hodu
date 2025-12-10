//! Plugin registry - storage and lookup for installed plugins

pub use super::types::{DetectedPluginType, PluginEntry, PluginType};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

/// Plugin registry stored in ~/.hodu/plugins.json
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginRegistry {
    /// Schema version for future compatibility
    pub version: u32,
    /// Installed plugins
    pub plugins: Vec<PluginEntry>,
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

    /// Load registry from file with shared lock
    pub fn load(path: &Path) -> Result<Self, RegistryError> {
        if !path.exists() {
            return Ok(Self::new());
        }

        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| RegistryError::Io(e.to_string()))?;

        // Acquire shared lock for reading
        file.lock_shared().map_err(|e| RegistryError::Lock(e.to_string()))?;

        let content = std::fs::read_to_string(path).map_err(|e| RegistryError::Io(e.to_string()))?;

        let registry: Self = serde_json::from_str(&content).map_err(|e| RegistryError::Parse(e.to_string()))?;

        // Lock released when file is dropped
        Ok(registry)
    }

    /// Save registry to file with exclusive lock
    pub fn save(&self, path: &Path) -> Result<(), RegistryError> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| RegistryError::Io(e.to_string()))?;
        }

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| RegistryError::Io(e.to_string()))?;

        // Acquire exclusive lock for writing
        file.lock_exclusive().map_err(|e| RegistryError::Lock(e.to_string()))?;

        let content = serde_json::to_string_pretty(self).map_err(|e| RegistryError::Serialize(e.to_string()))?;

        std::fs::write(path, content).map_err(|e| RegistryError::Io(e.to_string()))?;

        // Lock released when file is dropped
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

    /// List all backend plugins (including disabled)
    pub fn all_backends(&self) -> impl Iterator<Item = &PluginEntry> {
        self.plugins.iter().filter(|p| p.plugin_type == PluginType::Backend)
    }

    /// List enabled backend plugins
    pub fn backends(&self) -> impl Iterator<Item = &PluginEntry> {
        self.plugins
            .iter()
            .filter(|p| p.plugin_type == PluginType::Backend && p.enabled)
    }

    /// List all model format plugins (including disabled)
    pub fn all_model_formats(&self) -> impl Iterator<Item = &PluginEntry> {
        self.plugins.iter().filter(|p| p.plugin_type == PluginType::ModelFormat)
    }

    /// List enabled model format plugins
    pub fn model_formats(&self) -> impl Iterator<Item = &PluginEntry> {
        self.plugins
            .iter()
            .filter(|p| p.plugin_type == PluginType::ModelFormat && p.enabled)
    }

    /// List all tensor format plugins (including disabled)
    pub fn all_tensor_formats(&self) -> impl Iterator<Item = &PluginEntry> {
        self.plugins
            .iter()
            .filter(|p| p.plugin_type == PluginType::TensorFormat)
    }

    /// List enabled tensor format plugins
    pub fn tensor_formats(&self) -> impl Iterator<Item = &PluginEntry> {
        self.plugins
            .iter()
            .filter(|p| p.plugin_type == PluginType::TensorFormat && p.enabled)
    }

    /// Enable a plugin by name
    pub fn enable(&mut self, name: &str) -> bool {
        if let Some(plugin) = self.find_mut(name) {
            plugin.enabled = true;
            true
        } else {
            false
        }
    }

    /// Disable a plugin by name
    pub fn disable(&mut self, name: &str) -> bool {
        if let Some(plugin) = self.find_mut(name) {
            plugin.enabled = false;
            true
        } else {
            false
        }
    }

    /// Check if all dependencies of a plugin are installed and enabled
    pub fn check_dependencies(&self, name: &str) -> Result<(), Vec<String>> {
        let plugin = match self.find(name) {
            Some(p) => p,
            None => return Ok(()),
        };

        let missing: Vec<String> = plugin
            .dependencies
            .iter()
            .filter(|dep| {
                self.find(dep).map(|p| !p.enabled).unwrap_or(true) // not found = missing
            })
            .cloned()
            .collect();

        if missing.is_empty() {
            Ok(())
        } else {
            Err(missing)
        }
    }

    /// Find backend plugin by device (enabled only)
    /// Device comparison is case-insensitive (normalized to lowercase)
    pub fn find_backend_by_device(&self, device: &str) -> Option<&PluginEntry> {
        let device_lower = device.to_lowercase();
        self.backends()
            .find(|p| p.capabilities.devices.iter().any(|d| d.to_lowercase() == device_lower))
    }

    /// Find model format plugin by extension (enabled only)
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

    /// Find tensor format plugin by extension (enabled only)
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

/// Registry errors
#[derive(Debug)]
pub enum RegistryError {
    NoHomeDir,
    Io(String),
    Lock(String),
    Parse(String),
    Serialize(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::NoHomeDir => write!(f, "Could not find home directory"),
            RegistryError::Io(e) => write!(f, "IO error: {}", e),
            RegistryError::Lock(e) => write!(f, "Lock error: {}", e),
            RegistryError::Parse(e) => write!(f, "Parse error: {}", e),
            RegistryError::Serialize(e) => write!(f, "Serialize error: {}", e),
        }
    }
}

impl std::error::Error for RegistryError {}

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
        let plugin_version = manifest["plugin_version"].as_str().unwrap_or("0.1.0").to_string();
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
                plugin_version,
            });
        } else if has_model_format {
            return Ok(DetectedPluginType::ModelFormat {
                name,
                version,
                plugin_version,
            });
        } else if has_tensor_format {
            return Ok(DetectedPluginType::TensorFormat {
                name,
                version,
                plugin_version,
            });
        } else {
            // Default to ModelFormat if type cannot be determined
            return Ok(DetectedPluginType::ModelFormat {
                name,
                version,
                plugin_version,
            });
        }
    }

    // If no manifest, try to spawn the plugin and initialize it
    // This is a simplified stub - full implementation would spawn the process
    Err(PluginDetectError::NotAPlugin)
}
