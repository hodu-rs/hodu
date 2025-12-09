//! Tensor data types for cross-plugin communication

use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Plugin data type enum (independent from hodu_core::DType for ABI stability)
///
/// This enum has fixed discriminant values to ensure ABI stability across
/// plugin versions. New types can be added with new discriminant values without
/// breaking existing plugins.
///
/// Note: This enum is `#[non_exhaustive]` - new types may be added in future versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[non_exhaustive]
pub enum PluginDType {
    /// Boolean
    Bool = 0,
    /// 8-bit float (E4M3)
    F8E4M3 = 1,
    /// 8-bit float (E5M2)
    F8E5M2 = 2,
    /// 16-bit bfloat
    BF16 = 3,
    /// 16-bit float
    F16 = 4,
    /// 32-bit float
    F32 = 5,
    /// 64-bit float
    F64 = 6,
    /// 8-bit unsigned integer
    U8 = 7,
    /// 16-bit unsigned integer
    U16 = 8,
    /// 32-bit unsigned integer
    U32 = 9,
    /// 64-bit unsigned integer
    U64 = 10,
    /// 8-bit signed integer
    I8 = 11,
    /// 16-bit signed integer
    I16 = 12,
    /// 32-bit signed integer
    I32 = 13,
    /// 64-bit signed integer
    I64 = 14,
}

impl PluginDType {
    /// Get size of this data type in bytes
    pub const fn size_in_bytes(&self) -> usize {
        match self {
            Self::Bool | Self::F8E4M3 | Self::F8E5M2 | Self::U8 | Self::I8 => 1,
            Self::BF16 | Self::F16 | Self::U16 | Self::I16 => 2,
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::F64 | Self::U64 | Self::I64 => 8,
        }
    }

    /// Get name of this data type
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::F8E4M3 => "f8e4m3",
            Self::F8E5M2 => "f8e5m2",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
        }
    }

    /// Check if this is a floating point type
    pub const fn is_float(&self) -> bool {
        matches!(
            self,
            Self::F8E4M3 | Self::F8E5M2 | Self::BF16 | Self::F16 | Self::F32 | Self::F64
        )
    }

    /// Check if this is an integer type
    pub const fn is_integer(&self) -> bool {
        matches!(
            self,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 | Self::I8 | Self::I16 | Self::I32 | Self::I64
        )
    }

    /// Check if this is a signed type
    pub const fn is_signed(&self) -> bool {
        matches!(
            self,
            Self::F8E4M3
                | Self::F8E5M2
                | Self::BF16
                | Self::F16
                | Self::F32
                | Self::F64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
        )
    }
}

impl FromStr for PluginDType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bool" => Ok(Self::Bool),
            "f8e4m3" => Ok(Self::F8E4M3),
            "f8e5m2" => Ok(Self::F8E5M2),
            "bf16" => Ok(Self::BF16),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            "u8" => Ok(Self::U8),
            "u16" => Ok(Self::U16),
            "u32" => Ok(Self::U32),
            "u64" => Ok(Self::U64),
            "i8" => Ok(Self::I8),
            "i16" => Ok(Self::I16),
            "i32" => Ok(Self::I32),
            "i64" => Ok(Self::I64),
            _ => Err(format!("Unknown dtype: {}", s)),
        }
    }
}

impl std::fmt::Display for PluginDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Raw tensor data for cross-plugin communication
///
/// This struct is used to pass tensor data between the CLI and plugins
/// without depending on the full Tensor type and registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    /// Raw bytes of tensor data
    pub data: Vec<u8>,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: PluginDType,
}

impl TensorData {
    /// Create new tensor data
    pub fn new(data: Vec<u8>, shape: Vec<usize>, dtype: PluginDType) -> Self {
        Self { data, shape, dtype }
    }

    /// Number of elements in the tensor
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Size of data in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Check if tensor data is valid (size matches shape * dtype)
    pub fn is_valid(&self) -> bool {
        let expected_size = self.numel() * self.dtype.size_in_bytes();
        self.data.len() == expected_size
    }

    /// Get tensor rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Check if tensor is scalar (rank 0)
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }
}
