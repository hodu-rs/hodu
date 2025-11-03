use super::{BinaryLogicalOpT, BinaryOpT};
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};
use num_traits::float::Float;

pub(crate) struct Add;
pub(crate) struct Div;
pub(crate) struct Mul;
pub(crate) struct Sub;
pub(crate) struct Pow;
pub(crate) struct Maximum;
pub(crate) struct Minimum;

macro_rules! binary_op {
    ($op:ident, $name: literal, $e: expr) => {
        impl BinaryOpT for $op {
            const NAME: &'static str = $name;

            #[inline]
            fn bool(_: bool, _: bool) -> bool {
                todo!("no binary function for bool")
            }
            #[inline]
            fn f8e4m3(v1: F8E4M3, v2: F8E4M3) -> F8E4M3 {
                $e(v1, v2)
            }
            #[inline]
            fn f8e5m2(v1: F8E5M2, v2: F8E5M2) -> F8E5M2 {
                $e(v1, v2)
            }
            #[inline]
            fn bf16(v1: bf16, v2: bf16) -> bf16 {
                $e(v1, v2)
            }
            #[inline]
            fn f16(v1: f16, v2: f16) -> f16 {
                $e(v1, v2)
            }
            #[inline]
            fn f32(v1: f32, v2: f32) -> f32 {
                $e(v1, v2)
            }
            #[inline]
            fn f64(v1: f64, v2: f64) -> f64 {
                $e(v1, v2)
            }
            #[inline]
            fn u8(v1: u8, v2: u8) -> u8 {
                $e(v1, v2)
            }
            #[inline]
            fn u16(v1: u16, v2: u16) -> u16 {
                $e(v1, v2)
            }
            #[inline]
            fn u32(v1: u32, v2: u32) -> u32 {
                $e(v1, v2)
            }
            #[inline]
            fn u64(v1: u64, v2: u64) -> u64 {
                $e(v1, v2)
            }
            #[inline]
            fn i8(v1: i8, v2: i8) -> i8 {
                $e(v1, v2)
            }
            #[inline]
            fn i16(v1: i16, v2: i16) -> i16 {
                $e(v1, v2)
            }
            #[inline]
            fn i32(v1: i32, v2: i32) -> i32 {
                $e(v1, v2)
            }
            #[inline]
            fn i64(v1: i64, v2: i64) -> i64 {
                $e(v1, v2)
            }
        }
    };
}

binary_op!(Add, "add", |v1, v2| v1 + v2);
binary_op!(Sub, "sub", |v1, v2| v1 - v2);
binary_op!(Mul, "mul", |v1, v2| v1 * v2);
binary_op!(Div, "div", |v1, v2| v1 / v2);
impl BinaryOpT for Pow {
    const NAME: &'static str = "pow";

    #[inline]
    fn bool(_: bool, _: bool) -> bool {
        todo!("no binary function for bool")
    }
    #[inline]
    fn f8e4m3(v1: F8E4M3, v2: F8E4M3) -> F8E4M3 {
        v1.powf(v2)
    }
    #[inline]
    fn f8e5m2(v1: F8E5M2, v2: F8E5M2) -> F8E5M2 {
        v1.powf(v2)
    }
    #[inline]
    fn bf16(v1: bf16, v2: bf16) -> bf16 {
        v1.powf(v2)
    }
    #[inline]
    fn f16(v1: f16, v2: f16) -> f16 {
        v1.powf(v2)
    }
    #[inline]
    fn f32(v1: f32, v2: f32) -> f32 {
        v1.powf(v2)
    }
    #[inline]
    fn f64(v1: f64, v2: f64) -> f64 {
        v1.powf(v2)
    }
    #[inline]
    fn u8(v1: u8, v2: u8) -> u8 {
        v1.pow(v2 as u32)
    }
    #[inline]
    fn u16(v1: u16, v2: u16) -> u16 {
        v1.pow(v2 as u32)
    }
    #[inline]
    fn u32(v1: u32, v2: u32) -> u32 {
        v1.pow(v2)
    }
    #[inline]
    fn u64(v1: u64, v2: u64) -> u64 {
        v1.pow(v2 as u32)
    }
    #[inline]
    fn i8(v1: i8, v2: i8) -> i8 {
        v1.pow(v2 as u32)
    }
    #[inline]
    fn i16(v1: i16, v2: i16) -> i16 {
        v1.pow(v2 as u32)
    }
    #[inline]
    fn i32(v1: i32, v2: i32) -> i32 {
        v1.pow(v2 as u32)
    }
    #[inline]
    fn i64(v1: i64, v2: i64) -> i64 {
        v1.pow(v2 as u32)
    }
}
binary_op!(Maximum, "maximum", |v1, v2| if v1 < v2 { v2 } else { v1 });
binary_op!(Minimum, "minimum", |v1, v2| if v1 > v2 { v2 } else { v1 });

pub(crate) struct LogicalAnd;
pub(crate) struct LogicalOr;
pub(crate) struct LogicalXor;

impl BinaryLogicalOpT for LogicalAnd {
    const NAME: &'static str = "logical_and";

    #[inline]
    fn bool(v1: bool, v2: bool) -> bool {
        v1 && v2
    }
    #[inline]
    fn f8e4m3(v1: F8E4M3, v2: F8E4M3) -> bool {
        (v1 != F8E4M3::ZERO) && (v2 != F8E4M3::ZERO)
    }
    #[inline]
    fn f8e5m2(v1: F8E5M2, v2: F8E5M2) -> bool {
        (v1 != F8E5M2::ZERO) && (v2 != F8E5M2::ZERO)
    }
    #[inline]
    fn bf16(v1: bf16, v2: bf16) -> bool {
        (v1 != bf16::ZERO) && (v2 != bf16::ZERO)
    }
    #[inline]
    fn f16(v1: f16, v2: f16) -> bool {
        (v1 != f16::ZERO) && (v2 != f16::ZERO)
    }
    #[inline]
    fn f32(v1: f32, v2: f32) -> bool {
        (v1 != 0.0) && (v2 != 0.0)
    }
    #[inline]
    fn f64(v1: f64, v2: f64) -> bool {
        (v1 != 0.0) && (v2 != 0.0)
    }
    #[inline]
    fn u8(v1: u8, v2: u8) -> bool {
        (v1 != 0) && (v2 != 0)
    }
    #[inline]
    fn u16(v1: u16, v2: u16) -> bool {
        (v1 != 0) && (v2 != 0)
    }
    #[inline]
    fn u32(v1: u32, v2: u32) -> bool {
        (v1 != 0) && (v2 != 0)
    }
    #[inline]
    fn u64(v1: u64, v2: u64) -> bool {
        (v1 != 0) && (v2 != 0)
    }
    #[inline]
    fn i8(v1: i8, v2: i8) -> bool {
        (v1 != 0) && (v2 != 0)
    }
    #[inline]
    fn i16(v1: i16, v2: i16) -> bool {
        (v1 != 0) && (v2 != 0)
    }
    #[inline]
    fn i32(v1: i32, v2: i32) -> bool {
        (v1 != 0) && (v2 != 0)
    }
    #[inline]
    fn i64(v1: i64, v2: i64) -> bool {
        (v1 != 0) && (v2 != 0)
    }
}
impl BinaryLogicalOpT for LogicalOr {
    const NAME: &'static str = "logical_or";

    #[inline]
    fn bool(v1: bool, v2: bool) -> bool {
        v1 || v2
    }
    #[inline]
    fn f8e4m3(v1: F8E4M3, v2: F8E4M3) -> bool {
        (v1 != F8E4M3::ZERO) || (v2 != F8E4M3::ZERO)
    }
    #[inline]
    fn f8e5m2(v1: F8E5M2, v2: F8E5M2) -> bool {
        (v1 != F8E5M2::ZERO) || (v2 != F8E5M2::ZERO)
    }
    #[inline]
    fn bf16(v1: bf16, v2: bf16) -> bool {
        (v1 != bf16::ZERO) || (v2 != bf16::ZERO)
    }
    #[inline]
    fn f16(v1: f16, v2: f16) -> bool {
        (v1 != f16::ZERO) || (v2 != f16::ZERO)
    }
    #[inline]
    fn f32(v1: f32, v2: f32) -> bool {
        (v1 != 0.0) || (v2 != 0.0)
    }
    #[inline]
    fn f64(v1: f64, v2: f64) -> bool {
        (v1 != 0.0) || (v2 != 0.0)
    }
    #[inline]
    fn u8(v1: u8, v2: u8) -> bool {
        (v1 != 0) || (v2 != 0)
    }
    #[inline]
    fn u16(v1: u16, v2: u16) -> bool {
        (v1 != 0) || (v2 != 0)
    }
    #[inline]
    fn u32(v1: u32, v2: u32) -> bool {
        (v1 != 0) || (v2 != 0)
    }
    #[inline]
    fn u64(v1: u64, v2: u64) -> bool {
        (v1 != 0) || (v2 != 0)
    }
    #[inline]
    fn i8(v1: i8, v2: i8) -> bool {
        (v1 != 0) || (v2 != 0)
    }
    #[inline]
    fn i16(v1: i16, v2: i16) -> bool {
        (v1 != 0) || (v2 != 0)
    }
    #[inline]
    fn i32(v1: i32, v2: i32) -> bool {
        (v1 != 0) || (v2 != 0)
    }
    #[inline]
    fn i64(v1: i64, v2: i64) -> bool {
        (v1 != 0) || (v2 != 0)
    }
}
impl BinaryLogicalOpT for LogicalXor {
    const NAME: &'static str = "logical_xor";

    #[inline]
    fn bool(v1: bool, v2: bool) -> bool {
        v1 ^ v2
    }
    #[inline]
    fn f8e4m3(v1: F8E4M3, v2: F8E4M3) -> bool {
        (v1 != F8E4M3::ZERO) ^ (v2 != F8E4M3::ZERO)
    }
    #[inline]
    fn f8e5m2(v1: F8E5M2, v2: F8E5M2) -> bool {
        (v1 != F8E5M2::ZERO) ^ (v2 != F8E5M2::ZERO)
    }
    #[inline]
    fn bf16(v1: bf16, v2: bf16) -> bool {
        (v1 != bf16::ZERO) ^ (v2 != bf16::ZERO)
    }
    #[inline]
    fn f16(v1: f16, v2: f16) -> bool {
        (v1 != f16::ZERO) ^ (v2 != f16::ZERO)
    }
    #[inline]
    fn f32(v1: f32, v2: f32) -> bool {
        (v1 != 0.0) ^ (v2 != 0.0)
    }
    #[inline]
    fn f64(v1: f64, v2: f64) -> bool {
        (v1 != 0.0) ^ (v2 != 0.0)
    }
    #[inline]
    fn u8(v1: u8, v2: u8) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
    #[inline]
    fn u16(v1: u16, v2: u16) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
    #[inline]
    fn u32(v1: u32, v2: u32) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
    #[inline]
    fn u64(v1: u64, v2: u64) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
    #[inline]
    fn i8(v1: i8, v2: i8) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
    #[inline]
    fn i16(v1: i16, v2: i16) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
    #[inline]
    fn i32(v1: i32, v2: i32) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
    #[inline]
    fn i64(v1: i64, v2: i64) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
}
