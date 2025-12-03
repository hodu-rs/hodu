use hodu_plugin_sdk::{
    BackendPlugin, BackendPluginHandle, BackendPluginMetadata, FormatPlugin, FormatPluginHandle, FormatPluginMetadata,
    FFI_PROTOCOL_VERSION, SDK_VERSION,
};
use libloading::Library;
use std::path::Path;

/// Loaded backend plugin with its library handle
pub struct LoadedBackendPlugin {
    _library: Library,
    plugin: &'static dyn BackendPlugin,
    destroy_fn: unsafe extern "C" fn(*mut BackendPluginHandle),
    handle: *mut BackendPluginHandle,
}

impl LoadedBackendPlugin {
    /// Load a backend plugin from a dynamic library
    pub fn load(path: &Path) -> Result<Self, PluginLoadError> {
        // Safety: Loading untrusted code - user is responsible for trusting the plugin
        let library = unsafe { Library::new(path) }.map_err(|e| PluginLoadError::LibraryLoad(e.to_string()))?;

        // Get metadata first to validate SDK version
        let metadata_fn: unsafe extern "C" fn() -> BackendPluginMetadata = unsafe {
            *library
                .get::<unsafe extern "C" fn() -> BackendPluginMetadata>(b"hodu_backend_plugin_metadata")
                .map_err(|e| PluginLoadError::SymbolNotFound(e.to_string()))?
        };

        let metadata = unsafe { metadata_fn() };

        // Check FFI protocol version first (ABI compatibility)
        if metadata.ffi_version != FFI_PROTOCOL_VERSION {
            return Err(PluginLoadError::IncompatibleFfiVersion {
                expected: FFI_PROTOCOL_VERSION,
                found: metadata.ffi_version,
            });
        }

        // Validate SDK version (API compatibility)
        let sdk_version = unsafe {
            std::ffi::CStr::from_ptr(metadata.sdk_version)
                .to_str()
                .map_err(|e| PluginLoadError::InvalidMetadata(e.to_string()))?
        };

        if !is_compatible_sdk_version(sdk_version) {
            return Err(PluginLoadError::IncompatibleSdkVersion {
                expected: SDK_VERSION.to_string(),
                found: sdk_version.to_string(),
            });
        }

        // Get create and destroy function pointers
        // We copy the function pointers out so we can move library into struct
        let create_fn: unsafe extern "C" fn() -> *mut BackendPluginHandle = unsafe {
            *library
                .get::<unsafe extern "C" fn() -> *mut BackendPluginHandle>(b"hodu_backend_plugin_create")
                .map_err(|e| PluginLoadError::SymbolNotFound(e.to_string()))?
        };

        let destroy_fn: unsafe extern "C" fn(*mut BackendPluginHandle) = unsafe {
            *library
                .get::<unsafe extern "C" fn(*mut BackendPluginHandle)>(b"hodu_backend_plugin_destroy")
                .map_err(|e| PluginLoadError::SymbolNotFound(e.to_string()))?
        };

        // Create plugin instance
        let handle = unsafe { create_fn() };
        if handle.is_null() {
            return Err(PluginLoadError::CreateFailed);
        }

        // Convert handle to trait object reference
        // Safety: We trust the plugin implements the contract correctly
        let plugin: &'static dyn BackendPlugin = unsafe { BackendPluginHandle::as_ref(handle) };

        Ok(Self {
            _library: library,
            plugin,
            destroy_fn,
            handle,
        })
    }

    /// Get reference to the plugin
    pub fn plugin(&self) -> &dyn BackendPlugin {
        self.plugin
    }
}

impl Drop for LoadedBackendPlugin {
    fn drop(&mut self) {
        unsafe {
            (self.destroy_fn)(self.handle);
        }
    }
}

/// Loaded format plugin with its library handle
pub struct LoadedFormatPlugin {
    _library: Library,
    plugin: &'static dyn FormatPlugin,
    destroy_fn: unsafe extern "C" fn(*mut FormatPluginHandle),
    handle: *mut FormatPluginHandle,
}

impl LoadedFormatPlugin {
    /// Load a format plugin from a dynamic library
    pub fn load(path: &Path) -> Result<Self, PluginLoadError> {
        // Safety: Loading untrusted code - user is responsible for trusting the plugin
        let library = unsafe { Library::new(path) }.map_err(|e| PluginLoadError::LibraryLoad(e.to_string()))?;

        // Get metadata first to validate SDK version
        let metadata_fn: unsafe extern "C" fn() -> FormatPluginMetadata = unsafe {
            *library
                .get::<unsafe extern "C" fn() -> FormatPluginMetadata>(b"hodu_format_plugin_metadata")
                .map_err(|e| PluginLoadError::SymbolNotFound(e.to_string()))?
        };

        let metadata = unsafe { metadata_fn() };

        // Check FFI protocol version first (ABI compatibility)
        if metadata.ffi_version != FFI_PROTOCOL_VERSION {
            return Err(PluginLoadError::IncompatibleFfiVersion {
                expected: FFI_PROTOCOL_VERSION,
                found: metadata.ffi_version,
            });
        }

        // Validate SDK version (API compatibility)
        let sdk_version = unsafe {
            std::ffi::CStr::from_ptr(metadata.sdk_version)
                .to_str()
                .map_err(|e| PluginLoadError::InvalidMetadata(e.to_string()))?
        };

        if !is_compatible_sdk_version(sdk_version) {
            return Err(PluginLoadError::IncompatibleSdkVersion {
                expected: SDK_VERSION.to_string(),
                found: sdk_version.to_string(),
            });
        }

        // Get create and destroy function pointers
        // We copy the function pointers out so we can move library into struct
        let create_fn: unsafe extern "C" fn() -> *mut FormatPluginHandle = unsafe {
            *library
                .get::<unsafe extern "C" fn() -> *mut FormatPluginHandle>(b"hodu_format_plugin_create")
                .map_err(|e| PluginLoadError::SymbolNotFound(e.to_string()))?
        };

        let destroy_fn: unsafe extern "C" fn(*mut FormatPluginHandle) = unsafe {
            *library
                .get::<unsafe extern "C" fn(*mut FormatPluginHandle)>(b"hodu_format_plugin_destroy")
                .map_err(|e| PluginLoadError::SymbolNotFound(e.to_string()))?
        };

        // Create plugin instance
        let handle = unsafe { create_fn() };
        if handle.is_null() {
            return Err(PluginLoadError::CreateFailed);
        }

        // Convert handle to trait object reference
        // Safety: We trust the plugin implements the contract correctly
        let plugin: &'static dyn FormatPlugin = unsafe { FormatPluginHandle::as_ref(handle) };

        Ok(Self {
            _library: library,
            plugin,
            destroy_fn,
            handle,
        })
    }

    /// Get reference to the plugin
    pub fn plugin(&self) -> &dyn FormatPlugin {
        self.plugin
    }
}

impl Drop for LoadedFormatPlugin {
    fn drop(&mut self) {
        unsafe {
            (self.destroy_fn)(self.handle);
        }
    }
}

/// Check if plugin SDK version is compatible with host
fn is_compatible_sdk_version(plugin_version: &str) -> bool {
    // For now, require exact major.minor match
    let host_parts: Vec<&str> = SDK_VERSION.split('.').collect();
    let plugin_parts: Vec<&str> = plugin_version.split('.').collect();

    if host_parts.len() < 2 || plugin_parts.len() < 2 {
        return false;
    }

    // Major and minor must match
    host_parts[0] == plugin_parts[0] && host_parts[1] == plugin_parts[1]
}

/// Detect plugin type by trying to load metadata
pub fn detect_plugin_type(path: &Path) -> Result<DetectedPluginType, PluginLoadError> {
    let library = unsafe { Library::new(path) }.map_err(|e| PluginLoadError::LibraryLoad(e.to_string()))?;

    // Try backend first
    let is_backend: bool = unsafe {
        library
            .get::<unsafe extern "C" fn() -> BackendPluginMetadata>(b"hodu_backend_plugin_metadata")
            .is_ok()
    };

    if is_backend {
        let metadata_fn: unsafe extern "C" fn() -> BackendPluginMetadata = unsafe {
            *library
                .get::<unsafe extern "C" fn() -> BackendPluginMetadata>(b"hodu_backend_plugin_metadata")
                .unwrap()
        };
        let metadata = unsafe { metadata_fn() };
        let name = unsafe {
            std::ffi::CStr::from_ptr(metadata.name)
                .to_str()
                .unwrap_or("unknown")
                .to_string()
        };
        let version = unsafe {
            std::ffi::CStr::from_ptr(metadata.version)
                .to_str()
                .unwrap_or("0.0.0")
                .to_string()
        };
        let sdk_version = unsafe {
            std::ffi::CStr::from_ptr(metadata.sdk_version)
                .to_str()
                .unwrap_or("0.0.0")
                .to_string()
        };
        return Ok(DetectedPluginType::Backend {
            name,
            version,
            sdk_version,
        });
    }

    // Try format
    let is_format: bool = unsafe {
        library
            .get::<unsafe extern "C" fn() -> FormatPluginMetadata>(b"hodu_format_plugin_metadata")
            .is_ok()
    };

    if is_format {
        let metadata_fn: unsafe extern "C" fn() -> FormatPluginMetadata = unsafe {
            *library
                .get::<unsafe extern "C" fn() -> FormatPluginMetadata>(b"hodu_format_plugin_metadata")
                .unwrap()
        };
        let metadata = unsafe { metadata_fn() };
        let name = unsafe {
            std::ffi::CStr::from_ptr(metadata.name)
                .to_str()
                .unwrap_or("unknown")
                .to_string()
        };
        let version = unsafe {
            std::ffi::CStr::from_ptr(metadata.version)
                .to_str()
                .unwrap_or("0.0.0")
                .to_string()
        };
        let sdk_version = unsafe {
            std::ffi::CStr::from_ptr(metadata.sdk_version)
                .to_str()
                .unwrap_or("0.0.0")
                .to_string()
        };
        return Ok(DetectedPluginType::Format {
            name,
            version,
            sdk_version,
        });
    }

    Err(PluginLoadError::NotAPlugin)
}

/// Detected plugin type with metadata
pub enum DetectedPluginType {
    Backend {
        name: String,
        version: String,
        sdk_version: String,
    },
    Format {
        name: String,
        version: String,
        sdk_version: String,
    },
}

/// Plugin load errors
#[derive(Debug)]
pub enum PluginLoadError {
    LibraryLoad(String),
    SymbolNotFound(String),
    InvalidMetadata(String),
    IncompatibleFfiVersion { expected: u32, found: u32 },
    IncompatibleSdkVersion { expected: String, found: String },
    CreateFailed,
    NotAPlugin,
}

impl std::fmt::Display for PluginLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginLoadError::LibraryLoad(e) => write!(f, "Failed to load library: {}", e),
            PluginLoadError::SymbolNotFound(e) => write!(f, "Symbol not found: {}", e),
            PluginLoadError::InvalidMetadata(e) => write!(f, "Invalid metadata: {}", e),
            PluginLoadError::IncompatibleFfiVersion { expected, found } => {
                write!(
                    f,
                    "Incompatible FFI protocol version: expected {}, found {} (plugin must be rebuilt)",
                    expected, found
                )
            },
            PluginLoadError::IncompatibleSdkVersion { expected, found } => {
                write!(f, "Incompatible SDK version: expected {}, found {}", expected, found)
            },
            PluginLoadError::CreateFailed => write!(f, "Failed to create plugin instance"),
            PluginLoadError::NotAPlugin => write!(f, "Library is not a valid hodu plugin"),
        }
    }
}

impl std::error::Error for PluginLoadError {}
