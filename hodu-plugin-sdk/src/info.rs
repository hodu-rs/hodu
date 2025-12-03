//! Plugin info.toml schema
//!
//! Example info.toml for backend plugin:
//! ```toml
//! [plugin]
//! name = "my-backend"
//! version = "0.1.0"
//! type = "backend"
//!
//! [backend]
//! runner = true
//! builder = false
//! devices = ["CPU"]
//! targets = []
//! ```
//!
//! Example info.toml for format plugin:
//! ```toml
//! [plugin]
//! name = "my-format"
//! version = "0.1.0"
//! type = "format"
//!
//! [format]
//! load_model = true
//! save_model = true
//! load_tensor = false
//! save_tensor = false
//! extensions = ["myf", "myformat"]
//! ```

use serde::Deserialize;

/// Root structure of info.toml
#[derive(Debug, Clone, Deserialize)]
pub struct PluginInfo {
    pub plugin: PluginMeta,
    #[serde(default)]
    pub backend: Option<BackendInfo>,
    #[serde(default)]
    pub format: Option<FormatInfo>,
}

/// [plugin] section
#[derive(Debug, Clone, Deserialize)]
pub struct PluginMeta {
    pub name: String,
    pub version: String,
    #[serde(rename = "type")]
    pub plugin_type: PluginInfoType,
}

/// Plugin type in info.toml
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PluginInfoType {
    Backend,
    Format,
}

/// [backend] section
#[derive(Debug, Clone, Deserialize)]
pub struct BackendInfo {
    #[serde(default)]
    pub runner: bool,
    #[serde(default)]
    pub builder: bool,
    #[serde(default)]
    pub devices: Vec<String>,
    #[serde(default)]
    pub targets: Vec<String>,
}

/// [format] section
#[derive(Debug, Clone, Deserialize)]
pub struct FormatInfo {
    #[serde(default)]
    pub load_model: bool,
    #[serde(default)]
    pub save_model: bool,
    #[serde(default)]
    pub load_tensor: bool,
    #[serde(default)]
    pub save_tensor: bool,
    #[serde(default)]
    pub extensions: Vec<String>,
}

impl PluginInfo {
    /// Parse info.toml content
    pub fn parse(content: &str) -> Result<Self, toml_edit::de::Error> {
        toml_edit::de::from_str(content)
    }

    /// Validate that the info.toml is consistent
    pub fn validate(&self) -> Result<(), String> {
        match self.plugin.plugin_type {
            PluginInfoType::Backend => {
                if self.backend.is_none() {
                    return Err("[backend] section is required for backend plugins".into());
                }
                if self.format.is_some() {
                    return Err("[format] section should not be present for backend plugins".into());
                }
            },
            PluginInfoType::Format => {
                if self.format.is_none() {
                    return Err("[format] section is required for format plugins".into());
                }
                if self.backend.is_some() {
                    return Err("[backend] section should not be present for format plugins".into());
                }
            },
        }
        Ok(())
    }
}

impl Default for BackendInfo {
    fn default() -> Self {
        Self {
            runner: true,
            builder: false,
            devices: vec!["CPU".into()],
            targets: Vec::new(),
        }
    }
}

impl Default for FormatInfo {
    fn default() -> Self {
        Self {
            load_model: true,
            save_model: false,
            load_tensor: false,
            save_tensor: false,
            extensions: Vec::new(),
        }
    }
}
