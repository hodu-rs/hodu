//! Plugin error types

use std::fmt;

/// Plugin error type
///
/// Note: This enum is `#[non_exhaustive]` - new error variants may be added in future versions.
/// Always include a wildcard pattern when matching.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PluginError {
    /// Capability not supported (e.g., runner, builder, load_model)
    NotSupported(String),
    /// Invalid input or argument
    InvalidInput(String),
    /// I/O error
    Io(String),
    /// Execution error
    Execution(String),
    /// Internal error
    Internal(String),
    /// Load error (failed to load file)
    Load(String),
    /// Save error (failed to save file)
    Save(String),
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSupported(msg) => write!(f, "not supported: {}", msg),
            Self::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
            Self::Io(msg) => write!(f, "io error: {}", msg),
            Self::Execution(msg) => write!(f, "execution error: {}", msg),
            Self::Internal(msg) => write!(f, "internal error: {}", msg),
            Self::Load(msg) => write!(f, "load error: {}", msg),
            Self::Save(msg) => write!(f, "save error: {}", msg),
        }
    }
}

impl std::error::Error for PluginError {}

impl From<std::io::Error> for PluginError {
    fn from(e: std::io::Error) -> Self {
        PluginError::Io(e.to_string())
    }
}

/// Result type for plugin operations
pub type PluginResult<T> = Result<T, PluginError>;
