use crate::{layer::compat::*, types::DType};
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};
use num_traits::float::Float;

#[cfg(feature = "serde")]
mod serde_impls {
    use super::*;
    use bincode::{
        de::Decoder,
        enc::Encoder,
        error::{DecodeError, EncodeError},
        BorrowDecode, Decode, Encode,
    };
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize_f8e4m3<S>(value: &F8E4M3, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        f32::from(*value).serialize(serializer)
    }

    pub fn deserialize_f8e4m3<'de, D>(deserializer: D) -> Result<F8E4M3, D::Error>
    where
        D: Deserializer<'de>,
    {
        let f = f32::deserialize(deserializer)?;
        Ok(F8E4M3::from(f))
    }

    #[cfg(feature = "f8e5m2")]
    pub fn serialize_f8e5m2<S>(value: &F8E5M2, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        f32::from(*value).serialize(serializer)
    }

    #[cfg(feature = "f8e5m2")]
    pub fn deserialize_f8e5m2<'de, D>(deserializer: D) -> Result<F8E5M2, D::Error>
    where
        D: Deserializer<'de>,
    {
        let f = f32::deserialize(deserializer)?;
        Ok(F8E5M2::from(f))
    }

    pub fn serialize_bf16<S>(value: &bf16, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        f32::from(*value).serialize(serializer)
    }

    pub fn deserialize_bf16<'de, D>(deserializer: D) -> Result<bf16, D::Error>
    where
        D: Deserializer<'de>,
    {
        let f = f32::deserialize(deserializer)?;
        Ok(bf16::from_f32(f))
    }

    pub fn serialize_f16<S>(value: &f16, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        f32::from(*value).serialize(serializer)
    }

    pub fn deserialize_f16<'de, D>(deserializer: D) -> Result<f16, D::Error>
    where
        D: Deserializer<'de>,
    {
        let f = f32::deserialize(deserializer)?;
        Ok(f16::from_f32(f))
    }

    impl Encode for Scalar {
        fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
            match self {
                Scalar::BOOL(v) => {
                    0u8.encode(encoder)?;
                    v.encode(encoder)
                },
                Scalar::F8E4M3(v) => {
                    1u8.encode(encoder)?;
                    f32::from(*v).encode(encoder)
                },
                #[cfg(feature = "f8e5m2")]
                Scalar::F8E5M2(v) => {
                    2u8.encode(encoder)?;
                    f32::from(*v).encode(encoder)
                },
                Scalar::BF16(v) => {
                    3u8.encode(encoder)?;
                    f32::from(*v).encode(encoder)
                },
                Scalar::F16(v) => {
                    4u8.encode(encoder)?;
                    f32::from(*v).encode(encoder)
                },
                Scalar::F32(v) => {
                    5u8.encode(encoder)?;
                    v.encode(encoder)
                },
                #[cfg(feature = "f64")]
                Scalar::F64(v) => {
                    6u8.encode(encoder)?;
                    v.encode(encoder)
                },
                Scalar::U8(v) => {
                    7u8.encode(encoder)?;
                    v.encode(encoder)
                },
                #[cfg(feature = "u16")]
                Scalar::U16(v) => {
                    8u8.encode(encoder)?;
                    v.encode(encoder)
                },

                Scalar::U32(v) => {
                    9u8.encode(encoder)?;
                    v.encode(encoder)
                },
                #[cfg(feature = "u64")]
                Scalar::U64(v) => {
                    10u8.encode(encoder)?;
                    v.encode(encoder)
                },
                Scalar::I8(v) => {
                    11u8.encode(encoder)?;
                    v.encode(encoder)
                },
                #[cfg(feature = "i16")]
                Scalar::I16(v) => {
                    12u8.encode(encoder)?;
                    v.encode(encoder)
                },
                Scalar::I32(v) => {
                    13u8.encode(encoder)?;
                    v.encode(encoder)
                },
                #[cfg(feature = "i64")]
                Scalar::I64(v) => {
                    14u8.encode(encoder)?;
                    v.encode(encoder)
                },
            }
        }
    }

    impl<Context> Decode<Context> for Scalar {
        fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
            let discriminant = u8::decode(decoder)?;
            match discriminant {
                0 => Ok(Scalar::BOOL(bool::decode(decoder)?)),
                1 => {
                    let f = f32::decode(decoder)?;
                    Ok(Scalar::F8E4M3(F8E4M3::from(f)))
                },
                #[cfg(feature = "f8e5m2")]
                2 => {
                    let f = f32::decode(decoder)?;
                    Ok(Scalar::F8E5M2(F8E5M2::from(f)))
                },
                3 => {
                    let f = f32::decode(decoder)?;
                    Ok(Scalar::BF16(bf16::from_f32(f)))
                },
                4 => {
                    let f = f32::decode(decoder)?;
                    Ok(Scalar::F16(f16::from_f32(f)))
                },
                5 => Ok(Scalar::F32(f32::decode(decoder)?)),
                #[cfg(feature = "f64")]
                6 => Ok(Scalar::F64(f64::decode(decoder)?)),
                7 => Ok(Scalar::U8(u8::decode(decoder)?)),
                #[cfg(feature = "u16")]
                8 => Ok(Scalar::U16(u16::decode(decoder)?)),

                9 => Ok(Scalar::U32(u32::decode(decoder)?)),
                #[cfg(feature = "u64")]
                10 => Ok(Scalar::U64(u64::decode(decoder)?)),
                11 => Ok(Scalar::I8(i8::decode(decoder)?)),
                #[cfg(feature = "i16")]
                12 => Ok(Scalar::I16(i16::decode(decoder)?)),
                13 => Ok(Scalar::I32(i32::decode(decoder)?)),
                #[cfg(feature = "i64")]
                14 => Ok(Scalar::I64(i64::decode(decoder)?)),
                _ => Err(DecodeError::UnexpectedVariant {
                    allowed: &bincode::error::AllowedEnumVariants::Range { min: 0, max: 14 },
                    found: discriminant as u32,
                    type_name: "Scalar",
                }),
            }
        }
    }

    impl<Context> BorrowDecode<'_, Context> for Scalar {
        fn borrow_decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
            Self::decode(decoder)
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Scalar {
    BOOL(bool),
    #[cfg_attr(
        feature = "serde",
        serde(
            serialize_with = "serde_impls::serialize_f8e4m3",
            deserialize_with = "serde_impls::deserialize_f8e4m3"
        )
    )]
    F8E4M3(F8E4M3),
    #[cfg(feature = "f8e5m2")]
    #[cfg_attr(
        feature = "serde",
        serde(
            serialize_with = "serde_impls::serialize_f8e5m2",
            deserialize_with = "serde_impls::deserialize_f8e5m2"
        )
    )]
    F8E5M2(F8E5M2),
    #[cfg_attr(
        feature = "serde",
        serde(
            serialize_with = "serde_impls::serialize_bf16",
            deserialize_with = "serde_impls::deserialize_bf16"
        )
    )]
    BF16(bf16),
    #[cfg_attr(
        feature = "serde",
        serde(
            serialize_with = "serde_impls::serialize_f16",
            deserialize_with = "serde_impls::deserialize_f16"
        )
    )]
    F16(f16),
    F32(f32),
    #[cfg(feature = "f64")]
    F64(f64),
    U8(u8),
    #[cfg(feature = "u16")]
    U16(u16),

    U32(u32),
    #[cfg(feature = "u64")]
    U64(u64),
    I8(i8),
    #[cfg(feature = "i16")]
    I16(i16),
    I32(i32),
    #[cfg(feature = "i64")]
    I64(i64),
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BOOL(v) => write!(f, "{v}"),
            Self::F8E4M3(v) => {
                if let Some(precision) = f.precision() {
                    write!(f, "{:.precision$}", v.to_f32(), precision = precision)
                } else {
                    write!(f, "{}", v.to_f32())
                }
            },
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(v) => {
                if let Some(precision) = f.precision() {
                    write!(f, "{:.precision$}", v.to_f32(), precision = precision)
                } else {
                    write!(f, "{}", v.to_f32())
                }
            },
            Self::BF16(v) => {
                if let Some(precision) = f.precision() {
                    write!(f, "{:.precision$}", v.to_f32(), precision = precision)
                } else {
                    write!(f, "{}", v.to_f32())
                }
            },
            Self::F16(v) => {
                if let Some(precision) = f.precision() {
                    write!(f, "{:.precision$}", v.to_f32(), precision = precision)
                } else {
                    write!(f, "{}", v.to_f32())
                }
            },
            Self::F32(v) => {
                if let Some(precision) = f.precision() {
                    write!(f, "{v:.precision$}")
                } else {
                    write!(f, "{v}")
                }
            },
            #[cfg(feature = "f64")]
            Self::F64(v) => {
                if let Some(precision) = f.precision() {
                    write!(f, "{v:.precision$}")
                } else {
                    write!(f, "{v}")
                }
            },
            Self::U8(v) => write!(f, "{v}"),
            #[cfg(feature = "u16")]
            Self::U16(v) => write!(f, "{v}"),

            Self::U32(v) => write!(f, "{v}"),
            #[cfg(feature = "u64")]
            Self::U64(v) => write!(f, "{v}"),
            Self::I8(v) => write!(f, "{v}"),
            #[cfg(feature = "i16")]
            Self::I16(v) => write!(f, "{v}"),
            Self::I32(v) => write!(f, "{v}"),
            #[cfg(feature = "i64")]
            Self::I64(v) => write!(f, "{v}"),
        }
    }
}

impl fmt::Debug for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BOOL(v) => write!(f, "Scalar(dtype=bool, value={v})"),
            Self::F8E4M3(v) => write!(f, "Scalar(dtype=f8e4m3, value={:.8})", v.to_f32()),
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(v) => write!(f, "Scalar(dtype=f8e5m2, value={:.8})", v.to_f32()),
            Self::BF16(v) => write!(f, "Scalar(dtype=bf16, value={:.8})", v.to_f32()),
            Self::F16(v) => write!(f, "Scalar(dtype=f16, value={:.8})", v.to_f32()),
            Self::F32(v) => write!(f, "Scalar(dtype=f32, value={v:.8})"),
            #[cfg(feature = "f64")]
            Self::F64(v) => write!(f, "Scalar(dtype=f64, value={v:.8})"),
            Self::U8(v) => write!(f, "Scalar(dtype=u8, value={v})"),
            #[cfg(feature = "u16")]
            Self::U16(v) => write!(f, "Scalar(dtype=u16, value={v})"),

            Self::U32(v) => write!(f, "Scalar(dtype=u32, value={v})"),
            #[cfg(feature = "u64")]
            Self::U64(v) => write!(f, "Scalar(dtype=u64, value={v})"),
            Self::I8(v) => write!(f, "Scalar(dtype=i8, value={v})"),
            #[cfg(feature = "i16")]
            Self::I16(v) => write!(f, "Scalar(dtype=i16, value={v})"),
            Self::I32(v) => write!(f, "Scalar(dtype=i32, value={v})"),
            #[cfg(feature = "i64")]
            Self::I64(v) => write!(f, "Scalar(dtype=i64, value={v})"),
        }
    }
}

impl Scalar {
    pub fn new<T: Into<Self>>(value: T) -> Self {
        value.into()
    }

    pub fn zero(dtype: DType) -> Self {
        match dtype {
            DType::BOOL => Self::BOOL(false),
            DType::F8E4M3 => Self::F8E4M3(F8E4M3::ZERO),
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => Self::F8E5M2(F8E5M2::ZERO),
            DType::BF16 => Self::BF16(bf16::ZERO),
            DType::F16 => Self::F16(f16::ZERO),
            DType::F32 => Self::F32(0.0),
            #[cfg(feature = "f64")]
            DType::F64 => Self::F64(0.0),
            DType::U8 => Self::U8(0),
            #[cfg(feature = "u16")]
            DType::U16 => Self::U16(0),

            DType::U32 => Self::U32(0),
            #[cfg(feature = "u64")]
            DType::U64 => Self::U64(0),
            DType::I8 => Self::I8(0),
            #[cfg(feature = "i16")]
            DType::I16 => Self::I16(0),
            DType::I32 => Self::I32(0),
            #[cfg(feature = "i64")]
            DType::I64 => Self::I64(0),
        }
    }

    pub fn one(dtype: DType) -> Self {
        match dtype {
            DType::BOOL => Self::BOOL(true),
            DType::F8E4M3 => Self::F8E4M3(F8E4M3::ONE),
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => Self::F8E5M2(F8E5M2::ONE),
            DType::BF16 => Self::BF16(bf16::ONE),
            DType::F16 => Self::F16(f16::ONE),
            DType::F32 => Self::F32(1.0),
            #[cfg(feature = "f64")]
            DType::F64 => Self::F64(1.0),
            DType::U8 => Self::U8(1),
            #[cfg(feature = "u16")]
            DType::U16 => Self::U16(1),

            DType::U32 => Self::U32(1),
            #[cfg(feature = "u64")]
            DType::U64 => Self::U64(1),
            DType::I8 => Self::I8(1),
            #[cfg(feature = "i16")]
            DType::I16 => Self::I16(1),
            DType::I32 => Self::I32(1),
            #[cfg(feature = "i64")]
            DType::I64 => Self::I64(1),
        }
    }

    pub fn from_f32(value: f32, dtype: DType) -> Self {
        match dtype {
            DType::BOOL => Self::BOOL(value != 0.0),
            DType::F8E4M3 => Self::F8E4M3(F8E4M3::from_f32(value)),
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => Self::F8E5M2(F8E5M2::from_f32(value)),
            DType::BF16 => Self::BF16(bf16::from_f32(value)),
            DType::F16 => Self::F16(f16::from_f32(value)),
            DType::F32 => Self::F32(value),
            #[cfg(feature = "f64")]
            DType::F64 => Self::F64(value as f64),
            DType::U8 => Self::U8(value as u8),
            #[cfg(feature = "u16")]
            DType::U16 => Self::U16(value as u16),

            DType::U32 => Self::U32(value as u32),
            #[cfg(feature = "u64")]
            DType::U64 => Self::U64(value as u64),
            DType::I8 => Self::I8(value as i8),
            #[cfg(feature = "i16")]
            DType::I16 => Self::I16(value as i16),
            DType::I32 => Self::I32(value as i32),
            #[cfg(feature = "i64")]
            DType::I64 => Self::I64(value as i64),
        }
    }

    #[inline]
    pub fn dtype(&self) -> DType {
        match self {
            Self::BOOL(_) => DType::BOOL,
            Self::F8E4M3(_) => DType::F8E4M3,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(_) => DType::F8E5M2,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            #[cfg(feature = "f64")]
            Self::F64(_) => DType::F64,
            Self::U8(_) => DType::U8,
            #[cfg(feature = "u16")]
            Self::U16(_) => DType::U16,

            Self::U32(_) => DType::U32,
            #[cfg(feature = "u64")]
            Self::U64(_) => DType::U64,
            Self::I8(_) => DType::I8,
            #[cfg(feature = "i16")]
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            #[cfg(feature = "i64")]
            Self::I64(_) => DType::I64,
        }
    }

    #[inline]
    pub fn to_dtype(&self, dtype: DType) -> Scalar {
        if self.dtype() == dtype {
            return *self;
        }

        match dtype {
            DType::BOOL => Scalar::BOOL(self.to_bool()),
            DType::F8E4M3 => Scalar::F8E4M3(F8E4M3::from_f32(self.to_f32())),
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => Scalar::F8E5M2(F8E5M2::from_f32(self.to_f32())),
            DType::BF16 => Scalar::BF16(bf16::from_f32(self.to_f32())),
            DType::F16 => Scalar::F16(f16::from_f32(self.to_f32())),
            DType::F32 => Scalar::F32(self.to_f32()),
            #[cfg(feature = "f64")]
            DType::F64 => Scalar::F64(self.to_f64()),
            DType::U8 => Scalar::U8(self.to_u8()),
            #[cfg(feature = "u16")]
            DType::U16 => Scalar::U16(self.to_u16()),

            DType::U32 => Scalar::U32(self.to_u32()),
            #[cfg(feature = "u64")]
            DType::U64 => Scalar::U64(self.to_u64()),
            DType::I8 => Scalar::I8(self.to_i8()),
            #[cfg(feature = "i16")]
            DType::I16 => Scalar::I16(self.to_i16()),
            DType::I32 => Scalar::I32(self.to_i32()),
            #[cfg(feature = "i64")]
            DType::I64 => Scalar::I64(self.to_i64()),
        }
    }

    #[inline]
    pub fn is_float(&self) -> bool {
        match self {
            Self::F8E4M3(_) | Self::BF16(_) | Self::F16(_) | Self::F32(_) => true,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(_) => true,
            #[cfg(feature = "f64")]
            Self::F64(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_int(&self) -> bool {
        match self {
            Self::BOOL(_) | Self::F8E4M3(_) | Self::BF16(_) | Self::F16(_) | Self::F32(_) => false,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(_) => false,
            #[cfg(feature = "f64")]
            Self::F64(_) => false,
            _ => true,
        }
    }

    #[inline]
    pub fn is_integer_value(&self) -> bool {
        if self.is_int() {
            return true;
        }

        match self {
            Self::F8E4M3(v) => {
                let f = v.to_f32();
                f == f.trunc()
            },
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(v) => {
                let f = v.to_f32();
                f == f.trunc()
            },
            Self::BF16(v) => {
                let f = v.to_f32();
                f == f.trunc()
            },
            Self::F16(v) => {
                let f = v.to_f32();
                f == f.trunc()
            },
            Self::F32(v) => *v == v.trunc(),
            #[cfg(feature = "f64")]
            Self::F64(v) => *v == v.trunc(),
            _ => false,
        }
    }

    #[inline]
    pub fn to_bool(&self) -> bool {
        match *self {
            Self::BOOL(x) => x,
            Self::F8E4M3(x) => x.to_f32() != 0.0,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(x) => x.to_f32() != 0.0,
            Self::BF16(x) => x.to_f32() != 0.0,
            Self::F16(x) => x.to_f32() != 0.0,
            Self::F32(x) => x != 0.0,
            #[cfg(feature = "f64")]
            Self::F64(x) => x != 0.0,
            Self::U8(x) => x != 0,
            #[cfg(feature = "u16")]
            Self::U16(x) => x != 0,

            Self::U32(x) => x != 0,
            #[cfg(feature = "u64")]
            Self::U64(x) => x != 0,
            Self::I8(x) => x != 0,
            #[cfg(feature = "i16")]
            Self::I16(x) => x != 0,
            Self::I32(x) => x != 0,
            #[cfg(feature = "i64")]
            Self::I64(x) => x != 0,
        }
    }

    #[inline]
    pub fn to_f8e4m3(&self) -> F8E4M3 {
        match *self {
            Self::BOOL(x) => F8E4M3::from_f32(if x { 1.0 } else { 0.0 }),
            Self::F8E4M3(x) => x,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(x) => F8E4M3::from_f32(x.to_f32()),
            Self::BF16(x) => F8E4M3::from_f32(x.to_f32()),
            Self::F16(x) => F8E4M3::from_f32(x.to_f32()),
            Self::F32(x) => F8E4M3::from_f32(x),
            #[cfg(feature = "f64")]
            Self::F64(x) => F8E4M3::from_f32(x as f32),
            Self::U8(x) => F8E4M3::from_f32(x as f32),
            #[cfg(feature = "u16")]
            Self::U16(x) => F8E4M3::from_f32(x as f32),

            Self::U32(x) => F8E4M3::from_f32(x as f32),
            #[cfg(feature = "u64")]
            Self::U64(x) => F8E4M3::from_f32(x as f32),
            Self::I8(x) => F8E4M3::from_f32(x as f32),
            #[cfg(feature = "i16")]
            Self::I16(x) => F8E4M3::from_f32(x as f32),
            Self::I32(x) => F8E4M3::from_f32(x as f32),
            #[cfg(feature = "i64")]
            Self::I64(x) => F8E4M3::from_f32(x as f32),
        }
    }

    #[inline]
    #[cfg(feature = "f8e5m2")]
    pub fn to_f8e5m2(&self) -> F8E5M2 {
        match *self {
            Self::BOOL(x) => F8E5M2::from_f32(if x { 1.0 } else { 0.0 }),
            Self::F8E4M3(x) => F8E5M2::from_f32(x.to_f32()),
            Self::F8E5M2(x) => x,
            Self::BF16(x) => F8E5M2::from_f32(x.to_f32()),
            Self::F16(x) => F8E5M2::from_f32(x.to_f32()),
            Self::F32(x) => F8E5M2::from_f32(x),
            #[cfg(feature = "f64")]
            Self::F64(x) => F8E5M2::from_f32(x as f32),
            Self::U8(x) => F8E5M2::from_f32(x as f32),
            #[cfg(feature = "u16")]
            Self::U16(x) => F8E5M2::from_f32(x as f32),

            Self::U32(x) => F8E5M2::from_f32(x as f32),
            #[cfg(feature = "u64")]
            Self::U64(x) => F8E5M2::from_f32(x as f32),
            Self::I8(x) => F8E5M2::from_f32(x as f32),
            #[cfg(feature = "i16")]
            Self::I16(x) => F8E5M2::from_f32(x as f32),
            Self::I32(x) => F8E5M2::from_f32(x as f32),
            #[cfg(feature = "i64")]
            Self::I64(x) => F8E5M2::from_f32(x as f32),
        }
    }

    #[inline]
    pub fn to_bf16(&self) -> bf16 {
        match *self {
            Self::BOOL(x) => bf16::from_f32(if x { 1.0 } else { 0.0 }),
            Self::F8E4M3(x) => bf16::from_f32(x.to_f32()),
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(x) => bf16::from_f32(x.to_f32()),
            Self::BF16(x) => x,
            Self::F16(x) => bf16::from_f32(x.to_f32()),
            Self::F32(x) => bf16::from_f32(x),
            #[cfg(feature = "f64")]
            Self::F64(x) => bf16::from_f32(x as f32),
            Self::U8(x) => bf16::from_f32(x as f32),
            #[cfg(feature = "u16")]
            Self::U16(x) => bf16::from_f32(x as f32),

            Self::U32(x) => bf16::from_f32(x as f32),
            #[cfg(feature = "u64")]
            Self::U64(x) => bf16::from_f32(x as f32),
            Self::I8(x) => bf16::from_f32(x as f32),
            #[cfg(feature = "i16")]
            Self::I16(x) => bf16::from_f32(x as f32),
            Self::I32(x) => bf16::from_f32(x as f32),
            #[cfg(feature = "i64")]
            Self::I64(x) => bf16::from_f32(x as f32),
        }
    }

    #[inline]
    pub fn to_f16(&self) -> f16 {
        match *self {
            Self::BOOL(x) => f16::from_f32(if x { 1.0 } else { 0.0 }),
            Self::F8E4M3(x) => f16::from_f32(x.to_f32()),
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(x) => f16::from_f32(x.to_f32()),
            Self::BF16(x) => f16::from_f32(x.to_f32()),
            Self::F16(x) => x,
            Self::F32(x) => f16::from_f32(x),
            #[cfg(feature = "f64")]
            Self::F64(x) => f16::from_f32(x as f32),
            Self::U8(x) => f16::from_f32(x as f32),
            #[cfg(feature = "u16")]
            Self::U16(x) => f16::from_f32(x as f32),

            Self::U32(x) => f16::from_f32(x as f32),
            #[cfg(feature = "u64")]
            Self::U64(x) => f16::from_f32(x as f32),
            Self::I8(x) => f16::from_f32(x as f32),
            #[cfg(feature = "i16")]
            Self::I16(x) => f16::from_f32(x as f32),
            Self::I32(x) => f16::from_f32(x as f32),
            #[cfg(feature = "i64")]
            Self::I64(x) => f16::from_f32(x as f32),
        }
    }

    #[inline]
    pub fn to_f32(&self) -> f32 {
        match *self {
            Self::BOOL(x) => {
                if x {
                    1.0
                } else {
                    0.0
                }
            },
            Self::F8E4M3(x) => x.to_f32(),
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(x) => x.to_f32(),
            Self::BF16(x) => x.to_f32(),
            Self::F16(x) => x.to_f32(),
            Self::F32(x) => x,
            #[cfg(feature = "f64")]
            Self::F64(x) => x as f32,
            Self::U8(x) => x as f32,
            #[cfg(feature = "u16")]
            Self::U16(x) => x as f32,

            Self::U32(x) => x as f32,
            #[cfg(feature = "u64")]
            Self::U64(x) => x as f32,
            Self::I8(x) => x as f32,
            #[cfg(feature = "i16")]
            Self::I16(x) => x as f32,
            Self::I32(x) => x as f32,
            #[cfg(feature = "i64")]
            Self::I64(x) => x as f32,
        }
    }

    #[inline]
    #[cfg(feature = "f64")]
    pub fn to_f64(&self) -> f64 {
        match *self {
            Self::BOOL(x) => {
                if x {
                    1.0
                } else {
                    0.0
                }
            },
            Self::F8E4M3(x) => x.to_f64(),
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(x) => x.to_f64(),
            Self::BF16(x) => x.to_f64(),
            Self::F16(x) => x.to_f64(),
            Self::F32(x) => x as f64,
            Self::F64(x) => x,
            Self::U8(x) => x as f64,
            #[cfg(feature = "u16")]
            Self::U16(x) => x as f64,

            Self::U32(x) => x as f64,
            #[cfg(feature = "u64")]
            Self::U64(x) => x as f64,
            Self::I8(x) => x as f64,
            #[cfg(feature = "i16")]
            Self::I16(x) => x as f64,
            Self::I32(x) => x as f64,
            #[cfg(feature = "i64")]
            Self::I64(x) => x as f64,
        }
    }

    #[inline]
    pub fn to_u8(&self) -> u8 {
        match *self {
            Self::BOOL(x) => {
                if x {
                    1
                } else {
                    0
                }
            },
            Self::F8E4M3(x) => x.to_f32().clamp(0.0, u8::MAX as f32) as u8,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(x) => x.to_f32().clamp(0.0, u8::MAX as f32) as u8,
            Self::BF16(x) => x.to_f32().clamp(0.0, u8::MAX as f32) as u8,
            Self::F16(x) => x.to_f32().clamp(0.0, u8::MAX as f32) as u8,
            Self::F32(x) => x.clamp(0.0, u8::MAX as f32) as u8,
            #[cfg(feature = "f64")]
            Self::F64(x) => x.clamp(0.0, u8::MAX as f64) as u8,
            Self::U8(x) => x,
            #[cfg(feature = "u16")]
            Self::U16(x) => x.min(u8::MAX as u16) as u8,

            Self::U32(x) => x.min(u8::MAX as u32) as u8,
            #[cfg(feature = "u64")]
            Self::U64(x) => x.min(u8::MAX as u64) as u8,
            Self::I8(x) => x.max(0) as u8,
            #[cfg(feature = "i16")]
            Self::I16(x) => x.clamp(0, u8::MAX as i16) as u8,
            Self::I32(x) => x.clamp(0, u8::MAX as i32) as u8,
            #[cfg(feature = "i64")]
            Self::I64(x) => x.clamp(0, u8::MAX as i64) as u8,
        }
    }

    #[inline]
    #[cfg(feature = "u16")]
    pub fn to_u16(&self) -> u16 {
        match *self {
            Self::BOOL(x) => {
                if x {
                    1
                } else {
                    0
                }
            },
            Self::F8E4M3(x) => x.to_f32().clamp(0.0, u16::MAX as f32) as u16,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(x) => x.to_f32().clamp(0.0, u16::MAX as f32) as u16,
            Self::BF16(x) => x.to_f32().clamp(0.0, u16::MAX as f32) as u16,
            Self::F16(x) => x.to_f32().clamp(0.0, u16::MAX as f32) as u16,
            Self::F32(x) => x.clamp(0.0, u16::MAX as f32) as u16,
            #[cfg(feature = "f64")]
            Self::F64(x) => x.clamp(0.0, u16::MAX as f64) as u16,
            Self::U8(x) => x as u16,
            #[cfg(feature = "u16")]
            Self::U16(x) => x,

            Self::U32(x) => x.min(u16::MAX as u32) as u16,
            #[cfg(feature = "u64")]
            Self::U64(x) => x.min(u16::MAX as u64) as u16,
            Self::I8(x) => x.max(0) as u16,
            #[cfg(feature = "i16")]
            Self::I16(x) => x.max(0) as u16,
            Self::I32(x) => x.clamp(0, u16::MAX as i32) as u16,
            #[cfg(feature = "i64")]
            Self::I64(x) => x.clamp(0, u16::MAX as i64) as u16,
        }
    }

    #[inline]
    pub fn to_u32(&self) -> u32 {
        match *self {
            Self::BOOL(x) => {
                if x {
                    1
                } else {
                    0
                }
            },
            Self::F8E4M3(x) => x.to_f32().clamp(0.0, u32::MAX as f32) as u32,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(x) => x.to_f32().clamp(0.0, u32::MAX as f32) as u32,
            Self::BF16(x) => x.to_f32().clamp(0.0, u32::MAX as f32) as u32,
            Self::F16(x) => x.to_f32().clamp(0.0, u32::MAX as f32) as u32,
            Self::F32(x) => x.clamp(0.0, u32::MAX as f32) as u32,
            #[cfg(feature = "f64")]
            Self::F64(x) => x.clamp(0.0, u32::MAX as f64) as u32,
            Self::U8(x) => x as u32,
            #[cfg(feature = "u16")]
            Self::U16(x) => x as u32,

            Self::U32(x) => x,
            #[cfg(feature = "u64")]
            Self::U64(x) => x.min(u32::MAX as u64) as u32,
            Self::I8(x) => x.max(0) as u32,
            #[cfg(feature = "i16")]
            Self::I16(x) => x.max(0) as u32,
            Self::I32(x) => x.max(0) as u32,
            #[cfg(feature = "i64")]
            Self::I64(x) => x.clamp(0, u32::MAX as i64) as u32,
        }
    }

    #[inline]
    #[cfg(feature = "u64")]
    pub fn to_u64(&self) -> u64 {
        match *self {
            Self::BOOL(x) => {
                if x {
                    1
                } else {
                    0
                }
            },
            Self::F8E4M3(x) => x.to_f64().max(0.0) as u64,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(x) => x.to_f64().max(0.0) as u64,
            Self::BF16(x) => x.to_f64().max(0.0) as u64,
            Self::F16(x) => x.to_f64().max(0.0) as u64,
            Self::F32(x) => x.max(0.0) as u64,
            #[cfg(feature = "f64")]
            Self::F64(x) => x.max(0.0) as u64,
            Self::U8(x) => x as u64,
            #[cfg(feature = "u16")]
            Self::U16(x) => x as u64,

            Self::U32(x) => x as u64,
            #[cfg(feature = "u64")]
            Self::U64(x) => x,
            Self::I8(x) => x.max(0) as u64,
            #[cfg(feature = "i16")]
            Self::I16(x) => x.max(0) as u64,
            Self::I32(x) => x.max(0) as u64,
            #[cfg(feature = "i64")]
            Self::I64(x) => x.max(0) as u64,
        }
    }

    #[inline]
    pub fn to_i8(&self) -> i8 {
        match *self {
            Self::BOOL(x) => {
                if x {
                    1
                } else {
                    0
                }
            },
            Self::F8E4M3(x) => x.to_f32().clamp(i8::MIN as f32, i8::MAX as f32) as i8,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(x) => x.to_f32().clamp(i8::MIN as f32, i8::MAX as f32) as i8,
            Self::BF16(x) => x.to_f32().clamp(i8::MIN as f32, i8::MAX as f32) as i8,
            Self::F16(x) => x.to_f32().clamp(i8::MIN as f32, i8::MAX as f32) as i8,
            Self::F32(x) => x.clamp(i8::MIN as f32, i8::MAX as f32) as i8,
            #[cfg(feature = "f64")]
            Self::F64(x) => x.clamp(i8::MIN as f64, i8::MAX as f64) as i8,
            Self::U8(x) => x.min(i8::MAX as u8) as i8,
            #[cfg(feature = "u16")]
            Self::U16(x) => x.min(i8::MAX as u16) as i8,

            Self::U32(x) => x.min(i8::MAX as u32) as i8,
            #[cfg(feature = "u64")]
            Self::U64(x) => x.min(i8::MAX as u64) as i8,
            Self::I8(x) => x,
            #[cfg(feature = "i16")]
            Self::I16(x) => x.clamp(i8::MIN as i16, i8::MAX as i16) as i8,
            Self::I32(x) => x.clamp(i8::MIN as i32, i8::MAX as i32) as i8,
            #[cfg(feature = "i64")]
            Self::I64(x) => x.clamp(i8::MIN as i64, i8::MAX as i64) as i8,
        }
    }

    #[inline]
    #[cfg(feature = "i16")]
    pub fn to_i16(&self) -> i16 {
        match *self {
            Self::BOOL(x) => {
                if x {
                    1
                } else {
                    0
                }
            },
            Self::F8E4M3(x) => x.to_f32().clamp(i16::MIN as f32, i16::MAX as f32) as i16,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(x) => x.to_f32().clamp(i16::MIN as f32, i16::MAX as f32) as i16,
            Self::BF16(x) => x.to_f32().clamp(i16::MIN as f32, i16::MAX as f32) as i16,
            Self::F16(x) => x.to_f32().clamp(i16::MIN as f32, i16::MAX as f32) as i16,
            Self::F32(x) => x.clamp(i16::MIN as f32, i16::MAX as f32) as i16,
            #[cfg(feature = "f64")]
            Self::F64(x) => x.clamp(i16::MIN as f64, i16::MAX as f64) as i16,
            Self::U8(x) => x as i16,
            #[cfg(feature = "u16")]
            Self::U16(x) => x.min(i16::MAX as u16) as i16,

            Self::U32(x) => x.min(i16::MAX as u32) as i16,
            #[cfg(feature = "u64")]
            Self::U64(x) => x.min(i16::MAX as u64) as i16,
            Self::I8(x) => x as i16,
            #[cfg(feature = "i16")]
            Self::I16(x) => x,
            Self::I32(x) => x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            #[cfg(feature = "i64")]
            Self::I64(x) => x.clamp(i16::MIN as i64, i16::MAX as i64) as i16,
        }
    }

    #[inline]
    pub fn to_i32(&self) -> i32 {
        match *self {
            Self::BOOL(x) => {
                if x {
                    1
                } else {
                    0
                }
            },
            Self::F8E4M3(x) => x.to_f32() as i32,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(x) => x.to_f32() as i32,
            Self::BF16(x) => x.to_f32() as i32,
            Self::F16(x) => x.to_f32() as i32,
            Self::F32(x) => x as i32,
            #[cfg(feature = "f64")]
            Self::F64(x) => x.clamp(i32::MIN as f64, i32::MAX as f64) as i32,
            Self::U8(x) => x as i32,
            #[cfg(feature = "u16")]
            Self::U16(x) => x as i32,

            Self::U32(x) => x.min(i32::MAX as u32) as i32,
            #[cfg(feature = "u64")]
            Self::U64(x) => x.min(i32::MAX as u64) as i32,
            Self::I8(x) => x as i32,
            #[cfg(feature = "i16")]
            Self::I16(x) => x as i32,
            Self::I32(x) => x,
            #[cfg(feature = "i64")]
            Self::I64(x) => x.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        }
    }

    #[inline]
    #[cfg(feature = "i64")]
    pub fn to_i64(&self) -> i64 {
        match *self {
            Self::BOOL(x) => {
                if x {
                    1
                } else {
                    0
                }
            },
            Self::F8E4M3(x) => x.to_f64() as i64,
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(x) => x.to_f64() as i64,
            Self::BF16(x) => x.to_f64() as i64,
            Self::F16(x) => x.to_f64() as i64,
            Self::F32(x) => x as i64,
            #[cfg(feature = "f64")]
            Self::F64(x) => x as i64,
            Self::U8(x) => x as i64,
            #[cfg(feature = "u16")]
            Self::U16(x) => x as i64,

            Self::U32(x) => x as i64,
            #[cfg(feature = "u64")]
            Self::U64(x) => x.min(i64::MAX as u64) as i64,
            Self::I8(x) => x as i64,
            #[cfg(feature = "i16")]
            Self::I16(x) => x as i64,
            Self::I32(x) => x as i64,
            #[cfg(feature = "i64")]
            Self::I64(x) => x,
        }
    }
}

impl From<bool> for Scalar {
    #[inline]
    fn from(x: bool) -> Self {
        Self::BOOL(x)
    }
}

impl From<F8E4M3> for Scalar {
    #[inline]
    fn from(x: F8E4M3) -> Self {
        Self::F8E4M3(x)
    }
}

impl From<F8E5M2> for Scalar {
    #[inline]
    fn from(x: F8E5M2) -> Self {
        #[cfg(feature = "f8e5m2")]
        {
            Self::F8E5M2(x)
        }
        #[cfg(not(feature = "f8e5m2"))]
        {
            Self::F32(x.to_f32())
        }
    }
}

impl From<bf16> for Scalar {
    #[inline]
    fn from(x: bf16) -> Self {
        Self::BF16(x)
    }
}

impl From<f16> for Scalar {
    #[inline]
    fn from(x: f16) -> Self {
        Self::F16(x)
    }
}

impl From<f32> for Scalar {
    #[inline]
    fn from(x: f32) -> Self {
        Self::F32(x)
    }
}

impl From<f64> for Scalar {
    #[inline]
    fn from(x: f64) -> Self {
        #[cfg(feature = "f64")]
        {
            Self::F64(x)
        }
        #[cfg(not(feature = "f64"))]
        {
            Self::F32(x as f32)
        }
    }
}

impl From<u8> for Scalar {
    #[inline]
    fn from(x: u8) -> Self {
        Self::U8(x)
    }
}

impl From<u16> for Scalar {
    #[inline]
    fn from(x: u16) -> Self {
        #[cfg(feature = "u16")]
        {
            Self::U16(x)
        }
        #[cfg(not(feature = "u16"))]
        {
            Self::I32(x as i32)
        }
    }
}

impl From<u32> for Scalar {
    #[inline]
    fn from(x: u32) -> Self {
        Self::U32(x)
    }
}

impl From<u64> for Scalar {
    #[inline]
    fn from(x: u64) -> Self {
        #[cfg(feature = "u64")]
        {
            Self::U64(x)
        }
        #[cfg(all(not(feature = "u64"), feature = "i64"))]
        {
            Self::I64(x as i64)
        }
        #[cfg(not(any(feature = "u64", feature = "i64")))]
        {
            Self::U32(x as u32)
        }
    }
}

impl From<usize> for Scalar {
    #[inline]
    fn from(x: usize) -> Self {
        #[cfg(feature = "u64")]
        {
            Self::U64(x as u64)
        }
        #[cfg(all(not(feature = "u64"), feature = "i64"))]
        {
            Self::I64(x as i64)
        }
        #[cfg(not(any(feature = "u64", feature = "i64")))]
        {
            Self::U32(x as u32)
        }
    }
}

impl From<i8> for Scalar {
    #[inline]
    fn from(x: i8) -> Self {
        Self::I8(x)
    }
}

impl From<i16> for Scalar {
    #[inline]
    fn from(x: i16) -> Self {
        #[cfg(feature = "i16")]
        {
            Self::I16(x)
        }
        #[cfg(not(feature = "i16"))]
        {
            Self::I32(x as i32)
        }
    }
}

impl From<i32> for Scalar {
    #[inline]
    fn from(x: i32) -> Self {
        Self::I32(x)
    }
}

impl From<i64> for Scalar {
    #[inline]
    fn from(x: i64) -> Self {
        #[cfg(feature = "i64")]
        {
            Self::I64(x)
        }
        #[cfg(not(feature = "i64"))]
        {
            Self::I32(x as i32)
        }
    }
}

impl From<isize> for Scalar {
    #[inline]
    fn from(x: isize) -> Self {
        #[cfg(feature = "i64")]
        {
            Self::I64(x as i64)
        }
        #[cfg(not(feature = "i64"))]
        {
            Self::I32(x as i32)
        }
    }
}

impl ops::Add for Scalar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::BOOL(a), Self::BOOL(b)) => Self::BOOL(a || b),
            (Self::U8(a), Self::U8(b)) => Self::U8(a + b),
            #[cfg(feature = "u16")]
            (Self::U16(a), Self::U16(b)) => Self::U16(a + b),

            (Self::U32(a), Self::U32(b)) => Self::U32(a + b),
            #[cfg(feature = "u64")]
            (Self::U64(a), Self::U64(b)) => Self::U64(a + b),
            (Self::I8(a), Self::I8(b)) => Self::I8(a + b),
            #[cfg(feature = "i16")]
            (Self::I16(a), Self::I16(b)) => Self::I16(a + b),
            (Self::I32(a), Self::I32(b)) => Self::I32(a + b),
            #[cfg(feature = "i64")]
            (Self::I64(a), Self::I64(b)) => Self::I64(a + b),
            (Self::F8E4M3(a), Self::F8E4M3(b)) => Self::F8E4M3(a + b),
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(a), Self::F8E5M2(b)) => Self::F8E5M2(a + b),
            (Self::BF16(a), Self::BF16(b)) => Self::BF16(a + b),
            (Self::F16(a), Self::F16(b)) => Self::F16(a + b),
            (Self::F32(a), Self::F32(b)) => Self::F32(a + b),
            #[cfg(feature = "f64")]
            (Self::F64(a), Self::F64(b)) => Self::F64(a + b),
            (lhs, rhs) => panic!("Cannot add {:?} and {:?}", lhs, rhs),
        }
    }
}

impl ops::Sub for Scalar {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::BOOL(a), Self::BOOL(b)) => Self::BOOL(a && !b),
            (Self::U8(a), Self::U8(b)) => Self::U8(a - b),
            #[cfg(feature = "u16")]
            (Self::U16(a), Self::U16(b)) => Self::U16(a - b),

            (Self::U32(a), Self::U32(b)) => Self::U32(a - b),
            #[cfg(feature = "u64")]
            (Self::U64(a), Self::U64(b)) => Self::U64(a - b),
            (Self::I8(a), Self::I8(b)) => Self::I8(a - b),
            #[cfg(feature = "i16")]
            (Self::I16(a), Self::I16(b)) => Self::I16(a - b),
            (Self::I32(a), Self::I32(b)) => Self::I32(a - b),
            #[cfg(feature = "i64")]
            (Self::I64(a), Self::I64(b)) => Self::I64(a - b),
            (Self::F8E4M3(a), Self::F8E4M3(b)) => Self::F8E4M3(a - b),
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(a), Self::F8E5M2(b)) => Self::F8E5M2(a - b),
            (Self::BF16(a), Self::BF16(b)) => Self::BF16(a - b),
            (Self::F16(a), Self::F16(b)) => Self::F16(a - b),
            (Self::F32(a), Self::F32(b)) => Self::F32(a - b),
            #[cfg(feature = "f64")]
            (Self::F64(a), Self::F64(b)) => Self::F64(a - b),
            (lhs, rhs) => panic!("Cannot subtract {:?} from {:?}", rhs, lhs),
        }
    }
}

impl ops::Mul for Scalar {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::BOOL(a), Self::BOOL(b)) => Self::BOOL(a && b),
            (Self::U8(a), Self::U8(b)) => Self::U8(a * b),
            #[cfg(feature = "u16")]
            (Self::U16(a), Self::U16(b)) => Self::U16(a * b),

            (Self::U32(a), Self::U32(b)) => Self::U32(a * b),
            #[cfg(feature = "u64")]
            (Self::U64(a), Self::U64(b)) => Self::U64(a * b),
            (Self::I8(a), Self::I8(b)) => Self::I8(a * b),
            #[cfg(feature = "i16")]
            (Self::I16(a), Self::I16(b)) => Self::I16(a * b),
            (Self::I32(a), Self::I32(b)) => Self::I32(a * b),
            #[cfg(feature = "i64")]
            (Self::I64(a), Self::I64(b)) => Self::I64(a * b),
            (Self::F8E4M3(a), Self::F8E4M3(b)) => Self::F8E4M3(a * b),
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(a), Self::F8E5M2(b)) => Self::F8E5M2(a * b),
            (Self::BF16(a), Self::BF16(b)) => Self::BF16(a * b),
            (Self::F16(a), Self::F16(b)) => Self::F16(a * b),
            (Self::F32(a), Self::F32(b)) => Self::F32(a * b),
            #[cfg(feature = "f64")]
            (Self::F64(a), Self::F64(b)) => Self::F64(a * b),
            (lhs, rhs) => panic!("Cannot multiply {:?} by {:?}", rhs, lhs),
        }
    }
}

impl ops::Div for Scalar {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::BOOL(a), Self::BOOL(b)) => Self::BOOL(a && b),
            (Self::U8(a), Self::U8(b)) => Self::U8(a / b),
            #[cfg(feature = "u16")]
            (Self::U16(a), Self::U16(b)) => Self::U16(a / b),

            (Self::U32(a), Self::U32(b)) => Self::U32(a / b),
            #[cfg(feature = "u64")]
            (Self::U64(a), Self::U64(b)) => Self::U64(a / b),
            (Self::I8(a), Self::I8(b)) => Self::I8(a / b),
            #[cfg(feature = "i16")]
            (Self::I16(a), Self::I16(b)) => Self::I16(a / b),
            (Self::I32(a), Self::I32(b)) => Self::I32(a / b),
            #[cfg(feature = "i64")]
            (Self::I64(a), Self::I64(b)) => Self::I64(a / b),
            (Self::F8E4M3(a), Self::F8E4M3(b)) => Self::F8E4M3(a / b),
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(a), Self::F8E5M2(b)) => Self::F8E5M2(a / b),
            (Self::BF16(a), Self::BF16(b)) => Self::BF16(a / b),
            (Self::F16(a), Self::F16(b)) => Self::F16(a / b),
            (Self::F32(a), Self::F32(b)) => Self::F32(a / b),
            #[cfg(feature = "f64")]
            (Self::F64(a), Self::F64(b)) => Self::F64(a / b),
            (lhs, rhs) => panic!("Cannot divide {:?} by {:?}", rhs, lhs),
        }
    }
}

impl ops::Rem for Scalar {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::BOOL(a), Self::BOOL(b)) => Self::BOOL(a && b),
            (Self::U8(a), Self::U8(b)) => Self::U8(a % b),
            #[cfg(feature = "u16")]
            (Self::U16(a), Self::U16(b)) => Self::U16(a % b),

            (Self::U32(a), Self::U32(b)) => Self::U32(a % b),
            #[cfg(feature = "u64")]
            (Self::U64(a), Self::U64(b)) => Self::U64(a % b),
            (Self::I8(a), Self::I8(b)) => Self::I8(a % b),
            #[cfg(feature = "i16")]
            (Self::I16(a), Self::I16(b)) => Self::I16(a % b),
            (Self::I32(a), Self::I32(b)) => Self::I32(a % b),
            #[cfg(feature = "i64")]
            (Self::I64(a), Self::I64(b)) => Self::I64(a % b),
            (Self::F8E4M3(a), Self::F8E4M3(b)) => Self::F8E4M3(a % b),
            #[cfg(feature = "f8e5m2")]
            (Self::F8E5M2(a), Self::F8E5M2(b)) => Self::F8E5M2(a % b),
            (Self::BF16(a), Self::BF16(b)) => Self::BF16(a % b),
            (Self::F16(a), Self::F16(b)) => Self::F16(a % b),
            (Self::F32(a), Self::F32(b)) => Self::F32(a % b),
            #[cfg(feature = "f64")]
            (Self::F64(a), Self::F64(b)) => Self::F64(a % b),
            (lhs, rhs) => panic!("Cannot modulo {:?} by {:?}", rhs, lhs),
        }
    }
}

impl Scalar {
    pub fn powi(&self, exp: i32) -> Self {
        match self {
            Self::BOOL(a) => {
                if exp == 0 {
                    Self::BOOL(true)
                } else {
                    Self::BOOL(*a)
                }
            },
            Self::U8(a) => Self::U8(a.pow(exp as u32)),
            #[cfg(feature = "u16")]
            Self::U16(a) => Self::U16(a.pow(exp as u32)),

            Self::U32(a) => Self::U32(a.pow(exp as u32)),
            #[cfg(feature = "u64")]
            Self::U64(a) => Self::U64(a.pow(exp as u32)),
            Self::I8(a) => Self::I8(a.pow(exp as u32)),
            #[cfg(feature = "i16")]
            Self::I16(a) => Self::I16(a.pow(exp as u32)),
            Self::I32(a) => Self::I32(a.pow(exp as u32)),
            #[cfg(feature = "i64")]
            Self::I64(a) => Self::I64(a.pow(exp as u32)),
            Self::F8E4M3(a) => Self::F8E4M3(a.powi(exp)),
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(a) => Self::F8E5M2(a.powi(exp)),
            Self::BF16(a) => Self::BF16(a.powi(exp)),
            Self::F16(a) => Self::F16(a.powi(exp)),
            Self::F32(a) => Self::F32(a.powi(exp)),
            #[cfg(feature = "f64")]
            Self::F64(a) => Self::F64(a.powi(exp)),
        }
    }

    pub fn sqrt(&self) -> Self {
        match self {
            Self::BOOL(a) => Self::BOOL(*a),
            Self::U8(a) => Scalar::U8(((*a as f32).sqrt()).clamp(0.0, u8::MAX as f32) as u8),
            #[cfg(feature = "u16")]
            Self::U16(a) => Scalar::U16(((*a as f32).sqrt()).clamp(0.0, u16::MAX as f32) as u16),

            Self::U32(a) => Scalar::U32(((*a as f32).sqrt()).clamp(0.0, u32::MAX as f32) as u32),
            #[cfg(feature = "u64")]
            Self::U64(a) => Scalar::U64(((*a as f32).sqrt()).clamp(0.0, u64::MAX as f32) as u64),
            Self::I8(a) => Scalar::I8(((*a as f32).sqrt()).clamp(0.0, i8::MAX as f32) as i8),
            #[cfg(feature = "i16")]
            Self::I16(a) => Scalar::I16(((*a as f32).sqrt()).clamp(0.0, i16::MAX as f32) as i16),
            Self::I32(a) => Scalar::I32(((*a as f32).sqrt()).clamp(0.0, i32::MAX as f32) as i32),
            #[cfg(feature = "i64")]
            Self::I64(a) => Scalar::I64(((*a as f32).sqrt()).clamp(0.0, i64::MAX as f32) as i64),
            Self::F8E4M3(a) => Self::F8E4M3(a.sqrt()),
            #[cfg(feature = "f8e5m2")]
            Self::F8E5M2(a) => Self::F8E5M2(a.sqrt()),
            Self::BF16(a) => Self::BF16(a.sqrt()),
            Self::F16(a) => Self::F16(a.sqrt()),
            Self::F32(a) => Self::F32(a.sqrt()),
            #[cfg(feature = "f64")]
            Self::F64(a) => Self::F64(a.sqrt()),
        }
    }
}
