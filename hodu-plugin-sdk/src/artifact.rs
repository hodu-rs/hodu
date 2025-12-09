//! Compiled artifact types for AOT compilation output

use crate::PluginDType;

/// Compiled artifact produced by a backend's build function
#[derive(Debug, Clone)]
pub struct CompiledArtifact {
    /// The format of this artifact (e.g., "sharedlib", "staticlib", "object", "metallib", "ptx")
    pub format: String,

    /// Target device this was compiled for (e.g., "cpu", "cuda::0", "metal")
    pub device: String,

    /// The compiled binary data
    pub data: Vec<u8>,

    /// Metadata about inputs
    pub inputs: Vec<ArtifactTensorInfo>,

    /// Metadata about outputs
    pub outputs: Vec<ArtifactTensorInfo>,

    /// Optional symbol table for native artifacts
    pub symbols: Option<ArtifactSymbols>,
}

/// Tensor metadata in compiled artifact
#[derive(Debug, Clone)]
pub struct ArtifactTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: PluginDType,
}

/// Symbol information for native artifacts
#[derive(Debug, Clone, Default)]
pub struct ArtifactSymbols {
    /// Entry point function name
    pub entry_point: String,
    /// Additional exported symbols
    pub exports: Vec<String>,
}

impl CompiledArtifact {
    /// Create a new compiled artifact
    pub fn new(format: impl Into<String>, device: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            format: format.into(),
            device: device.into(),
            data,
            inputs: Vec::new(),
            outputs: Vec::new(),
            symbols: None,
        }
    }

    /// Add input tensor info
    pub fn with_input(mut self, name: impl Into<String>, shape: Vec<usize>, dtype: PluginDType) -> Self {
        self.inputs.push(ArtifactTensorInfo {
            name: name.into(),
            shape,
            dtype,
        });
        self
    }

    /// Add output tensor info
    pub fn with_output(mut self, name: impl Into<String>, shape: Vec<usize>, dtype: PluginDType) -> Self {
        self.outputs.push(ArtifactTensorInfo {
            name: name.into(),
            shape,
            dtype,
        });
        self
    }

    /// Set symbol information
    pub fn with_symbols(mut self, symbols: ArtifactSymbols) -> Self {
        self.symbols = Some(symbols);
        self
    }

    /// Get the raw data
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get data size in bytes
    pub fn size(&self) -> usize {
        self.data.len()
    }
}
