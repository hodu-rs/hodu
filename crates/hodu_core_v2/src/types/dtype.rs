#![allow(non_upper_case_globals)]

use crate::layer::compat::*;

pub const bool: DType = DType::BOOL;
pub const f8e4m3: DType = DType::F8E4M3;
#[cfg(feature = "f8e5m2")]
pub const f8e5m2: DType = DType::F8E5M2;
pub const bfloat16: DType = DType::BF16;
pub const bf16: DType = DType::BF16;
pub const float16: DType = DType::F16;
pub const f16: DType = DType::F16;
pub const half: DType = DType::F16;
pub const float32: DType = DType::F32;
pub const f32: DType = DType::F32;
#[cfg(feature = "f64")]
pub const float64: DType = DType::F64;
#[cfg(feature = "f64")]
pub const f64: DType = DType::F64;
pub const uint8: DType = DType::U8;
pub const u8: DType = DType::U8;
#[cfg(feature = "u16")]
pub const uint16: DType = DType::U16;
#[cfg(feature = "u16")]
pub const u16: DType = DType::U16;
#[cfg(feature = "u32")]
pub const uint32: DType = DType::U32;
#[cfg(feature = "u32")]
pub const u32: DType = DType::U32;
#[cfg(feature = "u64")]
pub const uint64: DType = DType::U64;
#[cfg(feature = "u64")]
pub const u64: DType = DType::U64;
pub const int8: DType = DType::I8;
pub const i8: DType = DType::I8;
#[cfg(feature = "i16")]
pub const int16: DType = DType::I16;
#[cfg(feature = "i16")]
pub const i16: DType = DType::I16;
pub const int32: DType = DType::I32;
pub const i32: DType = DType::I32;
#[cfg(feature = "i64")]
pub const int64: DType = DType::I64;
#[cfg(feature = "i64")]
pub const i64: DType = DType::I64;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum DType {
    BOOL,
    F8E4M3,
    #[cfg(feature = "f8e5m2")]
    F8E5M2,
    BF16,
    F16,
    F32,
    #[cfg(feature = "f64")]
    F64,
    U8,
    #[cfg(feature = "u16")]
    U16,
    #[cfg(feature = "u32")]
    U32,
    #[cfg(feature = "u64")]
    U64,
    I8,
    #[cfg(feature = "i16")]
    I16,
    I32,
    #[cfg(feature = "i64")]
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
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2 => "f8e5m2",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            #[cfg(feature = "f64")]
            Self::F64 => "f64",
            Self::U8 => "u8",
            #[cfg(feature = "u16")]
            Self::U16 => "u16",
            #[cfg(feature = "u32")]
            Self::U32 => "u32",
            #[cfg(feature = "u64")]
            Self::U64 => "u64",
            Self::I8 => "i8",
            #[cfg(feature = "i16")]
            Self::I16 => "i16",
            Self::I32 => "i32",
            #[cfg(feature = "i64")]
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
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2 => 1,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
            #[cfg(feature = "f64")]
            Self::F64 => 8,
            Self::U8 => 1,
            #[cfg(feature = "u16")]
            Self::U16 => 2,
            #[cfg(feature = "u32")]
            Self::U32 => 4,
            #[cfg(feature = "u64")]
            Self::U64 => 8,
            Self::I8 => 1,
            #[cfg(feature = "i16")]
            Self::I16 => 2,
            Self::I32 => 4,
            #[cfg(feature = "i64")]
            Self::I64 => 8,
        }
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, DType::BOOL)
    }

    pub fn is_float(&self) -> bool {
        match self {
            Self::F8E4M3 | Self::BF16 | Self::F16 | Self::F32 => true,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2 => true,
            #[cfg(feature = "f64")]
            Self::F64 => true,
            _ => false,
        }
    }

    pub fn is_uint(&self) -> bool {
        match self {
            Self::U8 => true,
            #[cfg(feature = "u16")]
            Self::U16 => true,
            #[cfg(feature = "u32")]
            Self::U32 => true,
            #[cfg(feature = "u64")]
            Self::U64 => true,
            _ => false,
        }
    }

    pub fn is_int(&self) -> bool {
        match self {
            Self::I8 => true,
            #[cfg(feature = "i16")]
            Self::I16 => true,
            Self::I32 => true,
            #[cfg(feature = "i64")]
            Self::I64 => true,
            _ => false,
        }
    }
}
