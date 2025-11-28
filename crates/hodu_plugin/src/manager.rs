//! Plugin manager for loading and managing plugins

#![allow(improper_ctypes_definitions)]

use crate::{CompilerPlugin, FormatPlugin, HoduError, HoduResult, RuntimePlugin};
use hodu_compat::*;
use std::path::{Path, PathBuf};

/// Loaded compiler plugin with its library handle
struct LoadedCompiler {
    plugin: Box<dyn CompilerPlugin>,
    #[allow(dead_code)]
    library: Option<libloading::Library>,
}

/// Loaded runtime plugin with its library handle
struct LoadedRuntime {
    plugin: Box<dyn RuntimePlugin>,
    #[allow(dead_code)]
    library: Option<libloading::Library>,
}

/// Loaded format plugin with its library handle
struct LoadedFormat {
    plugin: Box<dyn FormatPlugin>,
    #[allow(dead_code)]
    library: Option<libloading::Library>,
}

/// Plugin manager for loading and managing compiler, runtime, and format plugins
pub struct PluginManager {
    compilers: HashMap<String, LoadedCompiler>,
    runtimes: HashMap<String, LoadedRuntime>,
    formats: HashMap<String, LoadedFormat>,
    plugin_dir: PathBuf,
}

impl PluginManager {
    /// Create a new plugin manager with the given plugin directory
    pub fn new(plugin_dir: impl Into<PathBuf>) -> Self {
        Self {
            compilers: HashMap::new(),
            runtimes: HashMap::new(),
            formats: HashMap::new(),
            plugin_dir: plugin_dir.into(),
        }
    }

    /// Create a new plugin manager with builtin plugins registered
    pub fn with_builtins(plugin_dir: impl Into<PathBuf>) -> Self {
        let mut manager = Self::new(plugin_dir);
        manager.register_builtins();
        manager
    }

    /// Register all builtin plugins
    pub fn register_builtins(&mut self) {
        // Register InterpRuntime as the default builtin runtime
        self.register_runtime(Box::new(crate::InterpRuntime::new()));
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

        Ok(Self::with_builtins(plugin_dir))
    }

    /// Get the plugin directory path
    pub fn plugin_dir(&self) -> &Path {
        &self.plugin_dir
    }

    // ========== Compiler Plugin Loading ==========

    /// Load a compiler plugin from a dynamic library file
    pub fn load_compiler(&mut self, path: impl AsRef<Path>) -> HoduResult<()> {
        let path = path.as_ref();

        let library = unsafe { libloading::Library::new(path) }
            .map_err(|e| HoduError::IoError(format!("Failed to load plugin: {}", e)))?;

        type CreateFn = unsafe extern "C" fn() -> *mut dyn CompilerPlugin;
        let create_fn: libloading::Symbol<CreateFn> = unsafe { library.get(b"hodu_compiler_plugin_create") }
            .map_err(|e| HoduError::IoError(format!("Plugin missing create function: {}", e)))?;

        let plugin_ptr = unsafe { create_fn() };
        let plugin: Box<dyn CompilerPlugin> = unsafe { Box::from_raw(plugin_ptr) };

        let name = plugin.name().to_string();

        self.compilers.insert(
            name,
            LoadedCompiler {
                plugin,
                library: Some(library),
            },
        );

        Ok(())
    }

    /// Register a builtin compiler plugin (doesn't need dynamic loading)
    pub fn register_compiler(&mut self, plugin: Box<dyn CompilerPlugin>) {
        let name = plugin.name().to_string();
        self.compilers.insert(name, LoadedCompiler { plugin, library: None });
    }

    /// Get a compiler plugin by name
    pub fn compiler(&self, name: &str) -> Option<&dyn CompilerPlugin> {
        self.compilers.get(name).map(|c| c.plugin.as_ref())
    }

    /// List all loaded compiler names
    pub fn compiler_names(&self) -> Vec<&str> {
        self.compilers.keys().map(|s| s.as_str()).collect()
    }

    /// List all loaded compilers
    pub fn compilers(&self) -> impl Iterator<Item = &dyn CompilerPlugin> {
        self.compilers.values().map(|c| c.plugin.as_ref())
    }

    // ========== Runtime Plugin Loading ==========

    /// Load a runtime plugin from a dynamic library file
    pub fn load_runtime(&mut self, path: impl AsRef<Path>) -> HoduResult<()> {
        let path = path.as_ref();

        let library = unsafe { libloading::Library::new(path) }
            .map_err(|e| HoduError::IoError(format!("Failed to load plugin: {}", e)))?;

        type CreateFn = unsafe extern "C" fn() -> *mut dyn RuntimePlugin;
        let create_fn: libloading::Symbol<CreateFn> = unsafe { library.get(b"hodu_runtime_plugin_create") }
            .map_err(|e| HoduError::IoError(format!("Plugin missing create function: {}", e)))?;

        let plugin_ptr = unsafe { create_fn() };
        let plugin: Box<dyn RuntimePlugin> = unsafe { Box::from_raw(plugin_ptr) };

        let name = plugin.name().to_string();

        self.runtimes.insert(
            name,
            LoadedRuntime {
                plugin,
                library: Some(library),
            },
        );

        Ok(())
    }

    /// Register a builtin runtime plugin (doesn't need dynamic loading)
    pub fn register_runtime(&mut self, plugin: Box<dyn RuntimePlugin>) {
        let name = plugin.name().to_string();
        self.runtimes.insert(name, LoadedRuntime { plugin, library: None });
    }

    /// Get a runtime plugin by name
    pub fn runtime(&self, name: &str) -> Option<&dyn RuntimePlugin> {
        self.runtimes.get(name).map(|r| r.plugin.as_ref())
    }

    /// List all loaded runtime names
    pub fn runtime_names(&self) -> Vec<&str> {
        self.runtimes.keys().map(|s| s.as_str()).collect()
    }

    /// List all loaded runtimes
    pub fn runtimes(&self) -> impl Iterator<Item = &dyn RuntimePlugin> {
        self.runtimes.values().map(|r| r.plugin.as_ref())
    }

    // ========== Format Plugin Loading ==========

    /// Load a format plugin from a dynamic library file
    pub fn load_format(&mut self, path: impl AsRef<Path>) -> HoduResult<()> {
        let path = path.as_ref();

        let library = unsafe { libloading::Library::new(path) }
            .map_err(|e| HoduError::IoError(format!("Failed to load plugin: {}", e)))?;

        type CreateFn = unsafe extern "C" fn() -> *mut dyn FormatPlugin;
        let create_fn: libloading::Symbol<CreateFn> = unsafe { library.get(b"hodu_format_plugin_create") }
            .map_err(|e| HoduError::IoError(format!("Plugin missing create function: {}", e)))?;

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

    /// Register a builtin format plugin (doesn't need dynamic loading)
    pub fn register_format(&mut self, plugin: Box<dyn FormatPlugin>) {
        let name = plugin.name().to_string();
        self.formats.insert(name, LoadedFormat { plugin, library: None });
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

    /// List all loaded format names
    pub fn format_names(&self) -> Vec<&str> {
        self.formats.keys().map(|s| s.as_str()).collect()
    }

    /// List all loaded formats
    pub fn formats(&self) -> impl Iterator<Item = &dyn FormatPlugin> {
        self.formats.values().map(|f| f.plugin.as_ref())
    }

    // ========== Bulk Loading ==========

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

            let filename = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

            if filename.contains("compiler") {
                if let Err(e) = self.load_compiler(&path) {
                    eprintln!("Warning: Failed to load compiler plugin {:?}: {}", path, e);
                }
            } else if filename.contains("runtime") {
                if let Err(e) = self.load_runtime(&path) {
                    eprintln!("Warning: Failed to load runtime plugin {:?}: {}", path, e);
                }
            } else if filename.contains("format") {
                if let Err(e) = self.load_format(&path) {
                    eprintln!("Warning: Failed to load format plugin {:?}: {}", path, e);
                }
            }
        }

        Ok(())
    }
}
