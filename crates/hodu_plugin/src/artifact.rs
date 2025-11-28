//! Compiled artifact types for passing between compiler and runtime

use crate::{Device, OutputFormat};
use hodu_compat::*;

/// Compiled artifact produced by a compiler
///
/// This represents the compiled output that can be:
/// - Passed directly to a runtime for execution
/// - Serialized to disk for AOT deployment
#[derive(Debug, Clone)]
pub struct CompiledArtifact {
    /// The format of this artifact
    pub format: OutputFormat,

    /// Target device this was compiled for
    pub device: Device,

    /// The compiled binary data
    pub data: Vec<u8>,

    /// Metadata about inputs (name -> shape, dtype)
    pub inputs: Vec<ArtifactTensorInfo>,

    /// Metadata about outputs (name -> shape, dtype)
    pub outputs: Vec<ArtifactTensorInfo>,

    /// Optional symbol table for native artifacts
    pub symbols: Option<ArtifactSymbols>,
}

/// Tensor metadata in compiled artifact
#[derive(Debug, Clone)]
pub struct ArtifactTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: ArtifactDType,
}

/// Data type in artifact (always includes all types for ABI stability)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ArtifactDType {
    Bool = 0,
    F8E4M3 = 1,
    F8E5M2 = 2,
    BF16 = 3,
    F16 = 4,
    F32 = 5,
    F64 = 6,
    U8 = 7,
    U16 = 8,
    U32 = 9,
    U64 = 10,
    I8 = 11,
    I16 = 12,
    I32 = 13,
    I64 = 14,
}

impl From<hodu_core::types::DType> for ArtifactDType {
    fn from(dtype: hodu_core::types::DType) -> Self {
        use hodu_core::types::DType;
        match dtype {
            DType::BOOL => ArtifactDType::Bool,
            DType::F8E4M3 => ArtifactDType::F8E4M3,
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => ArtifactDType::F8E5M2,
            DType::BF16 => ArtifactDType::BF16,
            DType::F16 => ArtifactDType::F16,
            DType::F32 => ArtifactDType::F32,
            #[cfg(feature = "f64")]
            DType::F64 => ArtifactDType::F64,
            DType::U8 => ArtifactDType::U8,
            #[cfg(feature = "u16")]
            DType::U16 => ArtifactDType::U16,
            DType::U32 => ArtifactDType::U32,
            #[cfg(feature = "u64")]
            DType::U64 => ArtifactDType::U64,
            DType::I8 => ArtifactDType::I8,
            #[cfg(feature = "i16")]
            DType::I16 => ArtifactDType::I16,
            DType::I32 => ArtifactDType::I32,
            #[cfg(feature = "i64")]
            DType::I64 => ArtifactDType::I64,
        }
    }
}

impl TryFrom<ArtifactDType> for hodu_core::types::DType {
    type Error = crate::HoduError;

    fn try_from(dtype: ArtifactDType) -> Result<Self, Self::Error> {
        #[allow(unused_imports)]
        use crate::HoduError;
        use hodu_core::types::DType;
        match dtype {
            ArtifactDType::Bool => Ok(DType::BOOL),
            ArtifactDType::F8E4M3 => Ok(DType::F8E4M3),
            #[cfg(feature = "f8e5m2")]
            ArtifactDType::F8E5M2 => Ok(DType::F8E5M2),
            #[cfg(not(feature = "f8e5m2"))]
            ArtifactDType::F8E5M2 => Err(HoduError::UnsupportedDType {
                dtype: DType::F32, // placeholder
                reason: "f8e5m2 not enabled".into(),
            }),
            ArtifactDType::BF16 => Ok(DType::BF16),
            ArtifactDType::F16 => Ok(DType::F16),
            ArtifactDType::F32 => Ok(DType::F32),
            #[cfg(feature = "f64")]
            ArtifactDType::F64 => Ok(DType::F64),
            #[cfg(not(feature = "f64"))]
            ArtifactDType::F64 => Err(HoduError::UnsupportedDType {
                dtype: DType::F32,
                reason: "f64 not enabled".into(),
            }),
            ArtifactDType::U8 => Ok(DType::U8),
            #[cfg(feature = "u16")]
            ArtifactDType::U16 => Ok(DType::U16),
            #[cfg(not(feature = "u16"))]
            ArtifactDType::U16 => Err(HoduError::UnsupportedDType {
                dtype: DType::U32,
                reason: "u16 not enabled".into(),
            }),
            ArtifactDType::U32 => Ok(DType::U32),
            #[cfg(feature = "u64")]
            ArtifactDType::U64 => Ok(DType::U64),
            #[cfg(not(feature = "u64"))]
            ArtifactDType::U64 => Err(HoduError::UnsupportedDType {
                dtype: DType::U32,
                reason: "u64 not enabled".into(),
            }),
            ArtifactDType::I8 => Ok(DType::I8),
            #[cfg(feature = "i16")]
            ArtifactDType::I16 => Ok(DType::I16),
            #[cfg(not(feature = "i16"))]
            ArtifactDType::I16 => Err(HoduError::UnsupportedDType {
                dtype: DType::I32,
                reason: "i16 not enabled".into(),
            }),
            ArtifactDType::I32 => Ok(DType::I32),
            #[cfg(feature = "i64")]
            ArtifactDType::I64 => Ok(DType::I64),
            #[cfg(not(feature = "i64"))]
            ArtifactDType::I64 => Err(HoduError::UnsupportedDType {
                dtype: DType::I32,
                reason: "i64 not enabled".into(),
            }),
        }
    }
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
    pub fn new(format: OutputFormat, device: Device, data: Vec<u8>) -> Self {
        Self {
            format,
            device,
            data,
            inputs: Vec::new(),
            outputs: Vec::new(),
            symbols: None,
        }
    }

    /// Add input tensor info
    pub fn with_input(mut self, name: impl Into<String>, shape: Vec<usize>, dtype: ArtifactDType) -> Self {
        self.inputs.push(ArtifactTensorInfo {
            name: name.into(),
            shape,
            dtype,
        });
        self
    }

    /// Add output tensor info
    pub fn with_output(mut self, name: impl Into<String>, shape: Vec<usize>, dtype: ArtifactDType) -> Self {
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
