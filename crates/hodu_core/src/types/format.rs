/// File format for tensor serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Format {
    /// Hodu Data Tensor - compact binary format (postcard)
    #[default]
    HDT,
    /// JSON - human-readable format
    JSON,
}

impl Format {
    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "hdt" => Some(Self::HDT),
            "json" => Some(Self::JSON),
            _ => None,
        }
    }

    /// Detect format from file path
    pub fn from_path(path: &std::path::Path) -> Option<Self> {
        path.extension().and_then(|e| e.to_str()).and_then(Self::from_extension)
    }

    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::HDT => "hdt",
            Self::JSON => "json",
        }
    }
}
