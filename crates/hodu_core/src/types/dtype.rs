#![allow(non_upper_case_globals)]

use crate::compat::*;

pub const bool: DType = DType::BOOL;
pub const f8e4m3: DType = DType::F8E4M3;
pub const f8e5m2: DType = DType::F8E5M2;
pub const bfloat16: DType = DType::BF16;
pub const bf16: DType = DType::BF16;
pub const float16: DType = DType::F16;
pub const f16: DType = DType::F16;
pub const half: DType = DType::F16;
pub const float32: DType = DType::F32;
pub const f32: DType = DType::F32;
pub const float64: DType = DType::F64;
pub const f64: DType = DType::F64;
pub const uint8: DType = DType::U8;
pub const u8: DType = DType::U8;
pub const uint16: DType = DType::U16;
pub const u16: DType = DType::U16;
pub const uint32: DType = DType::U32;
pub const u32: DType = DType::U32;
pub const uint64: DType = DType::U64;
pub const u64: DType = DType::U64;
pub const int8: DType = DType::I8;
pub const i8: DType = DType::I8;
pub const int16: DType = DType::I16;
pub const i16: DType = DType::I16;
pub const int32: DType = DType::I32;
pub const i32: DType = DType::I32;
pub const int64: DType = DType::I64;
pub const i64: DType = DType::I64;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum DType {
    BOOL,
    F8E4M3,
    F8E5M2,
    BF16,
    F16,
    F32,
    F64,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
}

impl Default for DType {
    fn default() -> Self {
        Self::F32
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::BOOL => "bool",
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
        };
        write!(f, "{s}")
    }
}

impl fmt::Debug for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl DType {
    pub fn get_size_in_bytes(&self) -> usize {
        match self {
            Self::BOOL => 1,
            Self::F8E4M3 => 1,
            Self::F8E5M2 => 1,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
        }
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, DType::BOOL)
    }

    pub fn is_float(&self) -> bool {
        matches!(
            self,
            Self::F8E4M3 | Self::F8E5M2 | Self::BF16 | Self::F16 | Self::F32 | Self::F64
        )
    }

    pub fn is_uint(&self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }
}
