use super::{CmpOpT, CmpScalarOpT};
use crate::scalar::Scalar;
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};

pub(crate) struct Eq;
pub(crate) struct Ne;
pub(crate) struct Lt;
pub(crate) struct Le;
pub(crate) struct Gt;
pub(crate) struct Ge;

macro_rules! cmp_op {
    ($op:ident, $name:literal, $e:expr) => {
        impl CmpOpT for $op {
            const NAME: &'static str = $name;

            #[inline]
            fn bool(v1: bool, v2: bool) -> bool {
                $e(v1, v2)
            }
            #[inline]
            fn f8e4m3(v1: F8E4M3, v2: F8E4M3) -> bool {
                $e(v1, v2)
            }
            #[inline]
            fn f8e5m2(v1: F8E5M2, v2: F8E5M2) -> bool {
                $e(v1, v2)
            }
            #[inline]
            fn bf16(v1: bf16, v2: bf16) -> bool {
                $e(v1, v2)
            }
            #[inline]
            fn f16(v1: f16, v2: f16) -> bool {
                $e(v1, v2)
            }
            #[inline]
            fn f32(v1: f32, v2: f32) -> bool {
                $e(v1, v2)
            }
            #[inline]
            fn f64(v1: f64, v2: f64) -> bool {
                $e(v1, v2)
            }
            #[inline]
            fn u8(v1: u8, v2: u8) -> bool {
                $e(v1, v2)
            }
            #[inline]
            fn u16(v1: u16, v2: u16) -> bool {
                $e(v1, v2)
            }
            #[inline]
            fn u32(v1: u32, v2: u32) -> bool {
                $e(v1, v2)
            }
            #[inline]
            fn u64(v1: u64, v2: u64) -> bool {
                $e(v1, v2)
            }
            #[inline]
            fn i8(v1: i8, v2: i8) -> bool {
                $e(v1, v2)
            }
            #[inline]
            fn i16(v1: i16, v2: i16) -> bool {
                $e(v1, v2)
            }
            #[inline]
            fn i32(v1: i32, v2: i32) -> bool {
                $e(v1, v2)
            }
            #[inline]
            fn i64(v1: i64, v2: i64) -> bool {
                $e(v1, v2)
            }
        }
    };
}

cmp_op!(Eq, "eq", |v1, v2| v1 == v2);
cmp_op!(Ne, "ne", |v1, v2| v1 != v2);
cmp_op!(Lt, "lt", |v1, v2| v1 < v2);
cmp_op!(Le, "le", |v1, v2| v1 <= v2);
cmp_op!(Gt, "gt", |v1, v2| v1 > v2);
cmp_op!(Ge, "ge", |v1, v2| v1 >= v2);

pub(crate) struct EqScalar;
pub(crate) struct NeScalar;
pub(crate) struct LtScalar;
pub(crate) struct LeScalar;
pub(crate) struct GtScalar;
pub(crate) struct GeScalar;

macro_rules! cmp_scalar_op {
    ($op:ident, $name:literal, $e:expr) => {
        impl CmpScalarOpT for $op {
            const NAME: &'static str = $name;

            #[inline]
            fn bool(v1: bool, scalar: Scalar) -> bool {
                $e(v1, scalar.to_bool())
            }
            #[inline]
            fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> bool {
                $e(v1, scalar.to_f8e4m3())
            }
            #[inline]
            fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> bool {
                $e(v1, scalar.to_f8e5m2())
            }
            #[inline]
            fn bf16(v1: bf16, scalar: Scalar) -> bool {
                $e(v1, scalar.to_bf16())
            }
            #[inline]
            fn f16(v1: f16, scalar: Scalar) -> bool {
                $e(v1, scalar.to_f16())
            }
            #[inline]
            fn f32(v1: f32, scalar: Scalar) -> bool {
                $e(v1, scalar.to_f32())
            }
            #[inline]
            fn f64(v1: f64, scalar: Scalar) -> bool {
                $e(v1, scalar.to_f64())
            }
            #[inline]
            fn u8(v1: u8, scalar: Scalar) -> bool {
                $e(v1, scalar.to_u8())
            }
            #[inline]
            fn u16(v1: u16, scalar: Scalar) -> bool {
                $e(v1, scalar.to_u16())
            }
            #[inline]
            fn u32(v1: u32, scalar: Scalar) -> bool {
                $e(v1, scalar.to_u32())
            }
            #[inline]
            fn u64(v1: u64, scalar: Scalar) -> bool {
                $e(v1, scalar.to_u64())
            }
            #[inline]
            fn i8(v1: i8, scalar: Scalar) -> bool {
                $e(v1, scalar.to_i8())
            }
            #[inline]
            fn i16(v1: i16, scalar: Scalar) -> bool {
                $e(v1, scalar.to_i16())
            }
            #[inline]
            fn i32(v1: i32, scalar: Scalar) -> bool {
                $e(v1, scalar.to_i32())
            }
            #[inline]
            fn i64(v1: i64, scalar: Scalar) -> bool {
                $e(v1, scalar.to_i64())
            }
        }
    };
}

cmp_scalar_op!(EqScalar, "eq_scalar", |v1, v2| v1 == v2);
cmp_scalar_op!(NeScalar, "ne_scalar", |v1, v2| v1 != v2);
cmp_scalar_op!(LtScalar, "lt_scalar", |v1, v2| v1 < v2);
cmp_scalar_op!(LeScalar, "le_scalar", |v1, v2| v1 <= v2);
cmp_scalar_op!(GtScalar, "gt_scalar", |v1, v2| v1 > v2);
cmp_scalar_op!(GeScalar, "ge_scalar", |v1, v2| v1 >= v2);
