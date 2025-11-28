//! Plugin manager for loading and managing plugins

#![allow(improper_ctypes_definitions)]

use crate::{BackendPlugin, FormatPlugin, HoduError, HoduResult};
use hodu_compat::*;
use std::path::{Path, PathBuf};

/// Loaded backend plugin with its library handle
struct LoadedBackend {
    plugin: Box<dyn BackendPlugin>,
    #[allow(dead_code)]
    library: Option<libloading::Library>,
}

/// Loaded format plugin with its library handle
struct LoadedFormat {
    plugin: Box<dyn FormatPlugin>,
    #[allow(dead_code)]
    library: Option<libloading::Library>,
}

/// Plugin manager for loading and managing backend and format plugins
pub struct PluginManager {
    backends: HashMap<String, LoadedBackend>,
    formats: HashMap<String, LoadedFormat>,
    plugin_dir: PathBuf,
}

impl PluginManager {
    /// Create a new plugin manager with the given plugin directory
    pub fn new(plugin_dir: impl Into<PathBuf>) -> Self {
        Self {
            backends: HashMap::new(),
            formats: HashMap::new(),
            plugin_dir: plugin_dir.into(),
        }
    }

    /// Create a plugin manager with the default plugin directory (~/.hodu/plugins)
    pub fn with_default_dir() -> HoduResult<Self> {
        let home = std::env::var("HOME").map_err(|_| HoduError::IoError("HOME environment variable not set".into()))?;
        let plugin_dir = PathBuf::from(home).join(".hodu").join("plugins");

        // Create directory if it doesn't exist
        if !plugin_dir.exists() {
            std::fs::create_dir_all(&plugin_dir)
                .map_err(|e| HoduError::IoError(format!("Failed to create plugin directory: {}", e)))?;
        }

        Ok(Self::new(plugin_dir))
    }

    /// Get the plugin directory path
    pub fn plugin_dir(&self) -> &Path {
        &self.plugin_dir
    }

    /// Load a backend plugin from a dynamic library file
    pub fn load_backend(&mut self, path: impl AsRef<Path>) -> HoduResult<()> {
        let path = path.as_ref();

        // Load the library
        let library = unsafe { libloading::Library::new(path) }
            .map_err(|e| HoduError::IoError(format!("Failed to load plugin: {}", e)))?;

        // Get the create function
        // Signature: extern "C" fn() -> *mut dyn BackendPlugin
        type CreateFn = unsafe extern "C" fn() -> *mut dyn BackendPlugin;
        let create_fn: libloading::Symbol<CreateFn> = unsafe { library.get(b"hodu_backend_plugin_create") }
            .map_err(|e| HoduError::IoError(format!("Plugin missing create function: {}", e)))?;

        // Create the plugin instance
        let plugin_ptr = unsafe { create_fn() };
        let plugin: Box<dyn BackendPlugin> = unsafe { Box::from_raw(plugin_ptr) };

        let name = plugin.name().to_string();

        self.backends.insert(
            name,
            LoadedBackend {
                plugin,
                library: Some(library),
            },
        );

        Ok(())
    }

    /// Load a format plugin from a dynamic library file
    pub fn load_format(&mut self, path: impl AsRef<Path>) -> HoduResult<()> {
        let path = path.as_ref();

        // Load the library
        let library = unsafe { libloading::Library::new(path) }
            .map_err(|e| HoduError::IoError(format!("Failed to load plugin: {}", e)))?;

        // Get the create function
        // Signature: extern "C" fn() -> *mut dyn FormatPlugin
        type CreateFn = unsafe extern "C" fn() -> *mut dyn FormatPlugin;
        let create_fn: libloading::Symbol<CreateFn> = unsafe { library.get(b"hodu_format_plugin_create") }
            .map_err(|e| HoduError::IoError(format!("Plugin missing create function: {}", e)))?;

        // Create the plugin instance
        let plugin_ptr = unsafe { create_fn() };
        let plugin: Box<dyn FormatPlugin> = unsafe { Box::from_raw(plugin_ptr) };

        let name = plugin.name().to_string();

        self.formats.insert(
            name,
            LoadedFormat {
                plugin,
                library: Some(library),
            },
        );

        Ok(())
    }

    /// Load all plugins from the plugin directory
    pub fn load_all(&mut self) -> HoduResult<()> {
        if !self.plugin_dir.exists() {
            return Ok(());
        }

        let entries = std::fs::read_dir(&self.plugin_dir)
            .map_err(|e| HoduError::IoError(format!("Failed to read plugin directory: {}", e)))?;

        for entry in entries {
            let entry = entry.map_err(|e| HoduError::IoError(format!("Failed to read directory entry: {}", e)))?;
            let path = entry.path();

            // Check if it's a dynamic library
            let ext = path.extension().and_then(|e| e.to_str());
            let is_dylib = matches!(ext, Some("so") | Some("dylib") | Some("dll"));

            if !is_dylib {
                continue;
            }

            // Try to load as backend first, then as format
            let filename = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

            if filename.contains("backend") {
                if let Err(e) = self.load_backend(&path) {
                    eprintln!("Warning: Failed to load backend plugin {:?}: {}", path, e);
                }
            } else if filename.contains("format") {
                if let Err(e) = self.load_format(&path) {
                    eprintln!("Warning: Failed to load format plugin {:?}: {}", path, e);
                }
            }
        }

        Ok(())
    }

    /// Get a backend plugin by name
    pub fn backend(&self, name: &str) -> Option<&dyn BackendPlugin> {
        self.backends.get(name).map(|b| b.plugin.as_ref())
    }

    /// Get a format plugin by name
    pub fn format(&self, name: &str) -> Option<&dyn FormatPlugin> {
        self.formats.get(name).map(|f| f.plugin.as_ref())
    }

    /// Get a format plugin by file extension
    pub fn format_for_extension(&self, ext: &str) -> Option<&dyn FormatPlugin> {
        self.formats
            .values()
            .find(|f| f.plugin.supports_extension(ext))
            .map(|f| f.plugin.as_ref())
    }

    /// List all loaded backend names
    pub fn backend_names(&self) -> Vec<&str> {
        self.backends.keys().map(|s| s.as_str()).collect()
    }

    /// List all loaded format names
    pub fn format_names(&self) -> Vec<&str> {
        self.formats.keys().map(|s| s.as_str()).collect()
    }

    /// List all loaded backends with their info
    pub fn backends(&self) -> impl Iterator<Item = &dyn BackendPlugin> {
        self.backends.values().map(|b| b.plugin.as_ref())
    }

    /// List all loaded formats with their info
    pub fn formats(&self) -> impl Iterator<Item = &dyn FormatPlugin> {
        self.formats.values().map(|f| f.plugin.as_ref())
    }

    /// Register a builtin backend plugin (doesn't need dynamic loading)
    pub fn register_backend(&mut self, plugin: Box<dyn BackendPlugin>) {
        let name = plugin.name().to_string();
        self.backends.insert(name, LoadedBackend { plugin, library: None });
    }

    /// Register a builtin format plugin (doesn't need dynamic loading)
    pub fn register_format(&mut self, plugin: Box<dyn FormatPlugin>) {
        let name = plugin.name().to_string();
        self.formats.insert(name, LoadedFormat { plugin, library: None });
    }
}
