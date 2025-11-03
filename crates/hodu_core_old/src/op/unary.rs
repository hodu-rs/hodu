use super::{UnaryLogicalOpT, UnaryOpT, UnaryScalarOpT};
use crate::scalar::Scalar;
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};
use num_traits::float::Float;

pub(crate) struct Neg;
pub(crate) struct Abs;
pub(crate) struct Sign;
pub(crate) struct Square;
pub(crate) struct Sqrt;
pub(crate) struct Recip;
pub(crate) struct Relu;
pub(crate) struct Sigmoid;
pub(crate) struct Tanh;
pub(crate) struct Gelu;
pub(crate) struct Softplus;
pub(crate) struct Silu;
pub(crate) struct Mish;
pub(crate) struct Sin;
pub(crate) struct Cos;
pub(crate) struct Tan;
pub(crate) struct Exp;
pub(crate) struct Exp2;
pub(crate) struct Exp10;
pub(crate) struct Ln;
pub(crate) struct Log2;
pub(crate) struct Log10;

macro_rules! unary_op {
    ($op:ident, $name: literal, $a: ident, $e: expr) => {
        impl UnaryOpT for $op {
            const NAME: &'static str = $name;

            #[inline]
            fn bool(_: bool) -> bool {
                todo!("no unary function for bool")
            }
            #[inline]
            fn f8e4m3($a: F8E4M3) -> F8E4M3 {
                $e
            }
            #[inline]
            fn f8e5m2($a: F8E5M2) -> F8E5M2 {
                $e
            }
            #[inline]
            fn bf16($a: bf16) -> bf16 {
                $e
            }
            #[inline]
            fn f16($a: f16) -> f16 {
                $e
            }
            #[inline]
            fn f32($a: f32) -> f32 {
                $e
            }
            #[inline]
            fn f64($a: f64) -> f64 {
                $e
            }
            #[inline]
            fn u8(_: u8) -> u8 {
                todo!("no unary function for u8")
            }
            #[inline]
            fn u16(_: u16) -> u16 {
                todo!("no unary function for u16")
            }
            #[inline]
            fn u32(_: u32) -> u32 {
                todo!("no unary function for u32")
            }
            #[inline]
            fn u64(_: u64) -> u64 {
                todo!("no unary function for u64")
            }
            #[inline]
            fn i8(_: i8) -> i8 {
                todo!("no unary function for i8")
            }
            #[inline]
            fn i16(_: i16) -> i16 {
                todo!("no unary function for i16")
            }
            #[inline]
            fn i32(_: i32) -> i32 {
                todo!("no unary function for i32")
            }
            #[inline]
            fn i64(_: i64) -> i64 {
                todo!("no unary function for i64")
            }
        }
    };
}

impl UnaryOpT for Neg {
    const NAME: &'static str = "neg";

    #[inline]
    fn bool(_: bool) -> bool {
        todo!("no unary function for bool")
    }
    #[inline]
    fn f8e4m3(v: F8E4M3) -> F8E4M3 {
        -v
    }
    #[inline]
    fn f8e5m2(v: F8E5M2) -> F8E5M2 {
        -v
    }
    #[inline]
    fn bf16(v: bf16) -> bf16 {
        -v
    }
    #[inline]
    fn f16(v: f16) -> f16 {
        -v
    }
    #[inline]
    fn f32(v: f32) -> f32 {
        -v
    }
    #[inline]
    fn f64(v: f64) -> f64 {
        -v
    }
    #[inline]
    fn u8(_: u8) -> u8 {
        todo!("no unary function for u8")
    }
    #[inline]
    fn u16(_: u16) -> u16 {
        todo!("no unary function for u16")
    }
    #[inline]
    fn u32(_: u32) -> u32 {
        todo!("no unary function for u32")
    }
    #[inline]
    fn u64(_: u64) -> u64 {
        todo!("no unary function for u64")
    }
    #[inline]
    fn i8(v: i8) -> i8 {
        -v
    }
    #[inline]
    fn i16(v: i16) -> i16 {
        -v
    }
    #[inline]
    fn i32(v: i32) -> i32 {
        -v
    }
    #[inline]
    fn i64(v: i64) -> i64 {
        -v
    }
}
impl UnaryOpT for Abs {
    const NAME: &'static str = "abs";

    #[inline]
    fn bool(_: bool) -> bool {
        todo!("no unary function for bool")
    }
    #[inline]
    fn f8e4m3(v: F8E4M3) -> F8E4M3 {
        v.abs()
    }
    #[inline]
    fn f8e5m2(v: F8E5M2) -> F8E5M2 {
        v.abs()
    }
    #[inline]
    fn bf16(v: bf16) -> bf16 {
        v.abs()
    }
    #[inline]
    fn f16(v: f16) -> f16 {
        v.abs()
    }
    #[inline]
    fn f32(v: f32) -> f32 {
        v.abs()
    }
    #[inline]
    fn f64(v: f64) -> f64 {
        v.abs()
    }
    #[inline]
    fn u8(_: u8) -> u8 {
        todo!("no unary function for u8")
    }
    #[inline]
    fn u16(_: u16) -> u16 {
        todo!("no unary function for u16")
    }
    #[inline]
    fn u32(_: u32) -> u32 {
        todo!("no unary function for u32")
    }
    #[inline]
    fn u64(_: u64) -> u64 {
        todo!("no unary function for u64")
    }
    #[inline]
    fn i8(v: i8) -> i8 {
        v.abs()
    }
    #[inline]
    fn i16(v: i16) -> i16 {
        v.abs()
    }
    #[inline]
    fn i32(v: i32) -> i32 {
        v.abs()
    }
    #[inline]
    fn i64(v: i64) -> i64 {
        v.abs()
    }
}
impl UnaryOpT for Sign {
    const NAME: &'static str = "sign";

    #[inline]
    fn bool(_: bool) -> bool {
        todo!("no unary function for bool")
    }
    #[inline]
    fn f8e4m3(v: F8E4M3) -> F8E4M3 {
        F8E4M3::from((v > F8E4M3::ZERO) as i8 as f32) - F8E4M3::from((v < F8E4M3::ZERO) as i8 as f32)
    }
    #[inline]
    fn f8e5m2(v: F8E5M2) -> F8E5M2 {
        F8E5M2::from((v > F8E5M2::ZERO) as i8 as f32) - F8E5M2::from((v < F8E5M2::ZERO) as i8 as f32)
    }
    #[inline]
    fn bf16(v: bf16) -> bf16 {
        bf16::from((v > bf16::ZERO) as i8) - bf16::from((v < bf16::ZERO) as i8)
    }
    #[inline]
    fn f16(v: f16) -> f16 {
        f16::from((v > f16::ZERO) as i8) - f16::from((v < f16::ZERO) as i8)
    }
    #[inline]
    fn f32(v: f32) -> f32 {
        f32::from(v > 0.) - f32::from(v < 0.)
    }
    #[inline]
    fn f64(v: f64) -> f64 {
        f64::from(v > 0.) - f64::from(v < 0.)
    }
    #[inline]
    fn u8(_: u8) -> u8 {
        todo!("no unary function for u8")
    }
    #[inline]
    fn u16(_: u16) -> u16 {
        todo!("no unary function for u16")
    }
    #[inline]
    fn u32(_: u32) -> u32 {
        todo!("no unary function for u32")
    }
    #[inline]
    fn u64(_: u64) -> u64 {
        todo!("no unary function for u64")
    }
    #[inline]
    fn i8(v: i8) -> i8 {
        (v > 0) as i8 - (v < 0) as i8
    }
    #[inline]
    fn i16(v: i16) -> i16 {
        (v > 0) as i16 - (v < 0) as i16
    }
    #[inline]
    fn i32(v: i32) -> i32 {
        (v > 0) as i32 - (v < 0) as i32
    }
    #[inline]
    fn i64(v: i64) -> i64 {
        (v > 0) as i64 - (v < 0) as i64
    }
}
impl UnaryOpT for Square {
    const NAME: &'static str = "square";

    #[inline]
    fn bool(_: bool) -> bool {
        todo!("no unary function for bool")
    }
    #[inline]
    fn f8e4m3(v: F8E4M3) -> F8E4M3 {
        v * v
    }
    #[inline]
    fn f8e5m2(v: F8E5M2) -> F8E5M2 {
        v * v
    }
    #[inline]
    fn bf16(v: bf16) -> bf16 {
        v * v
    }
    #[inline]
    fn f16(v: f16) -> f16 {
        v * v
    }
    #[inline]
    fn f32(v: f32) -> f32 {
        v * v
    }
    #[inline]
    fn f64(v: f64) -> f64 {
        v * v
    }
    #[inline]
    fn u8(v: u8) -> u8 {
        v * v
    }
    #[inline]
    fn u16(v: u16) -> u16 {
        v * v
    }
    #[inline]
    fn u32(v: u32) -> u32 {
        v * v
    }
    #[inline]
    fn u64(v: u64) -> u64 {
        v * v
    }
    #[inline]
    fn i8(v: i8) -> i8 {
        v * v
    }
    #[inline]
    fn i16(v: i16) -> i16 {
        v * v
    }
    #[inline]
    fn i32(v: i32) -> i32 {
        v * v
    }
    #[inline]
    fn i64(v: i64) -> i64 {
        v * v
    }
}
unary_op!(Sqrt, "sqrt", v, v.sqrt());
unary_op!(Recip, "recip", v, v.recip());

impl UnaryOpT for Relu {
    const NAME: &'static str = "relu";

    #[inline]
    fn bool(_: bool) -> bool {
        todo!("no unary function for bool")
    }
    #[inline]
    fn f8e4m3(v: F8E4M3) -> F8E4M3 {
        if v > F8E4M3::ZERO {
            v
        } else {
            F8E4M3::ZERO
        }
    }
    #[inline]
    fn f8e5m2(v: F8E5M2) -> F8E5M2 {
        if v > F8E5M2::ZERO {
            v
        } else {
            F8E5M2::ZERO
        }
    }
    #[inline]
    fn bf16(v: bf16) -> bf16 {
        if v > bf16::ZERO {
            v
        } else {
            bf16::ZERO
        }
    }
    #[inline]
    fn f16(v: f16) -> f16 {
        if v > f16::ZERO {
            v
        } else {
            f16::ZERO
        }
    }
    #[inline]
    fn f32(v: f32) -> f32 {
        if v > 0.0 {
            v
        } else {
            0.0
        }
    }
    #[inline]
    fn f64(v: f64) -> f64 {
        if v > 0.0 {
            v
        } else {
            0.0
        }
    }
    #[inline]
    fn u8(_: u8) -> u8 {
        todo!("no unary function for u8")
    }
    #[inline]
    fn u16(_: u16) -> u16 {
        todo!("no unary function for u16")
    }
    #[inline]
    fn u32(_: u32) -> u32 {
        todo!("no unary function for u32")
    }
    #[inline]
    fn u64(_: u64) -> u64 {
        todo!("no unary function for u64")
    }
    #[inline]
    fn i8(v: i8) -> i8 {
        if v > 0 {
            v
        } else {
            0
        }
    }
    #[inline]
    fn i16(v: i16) -> i16 {
        if v > 0 {
            v
        } else {
            0
        }
    }
    #[inline]
    fn i32(v: i32) -> i32 {
        if v > 0 {
            v
        } else {
            0
        }
    }
    #[inline]
    fn i64(v: i64) -> i64 {
        if v > 0 {
            v
        } else {
            0
        }
    }
}
unary_op!(Sigmoid, "sigmoid", v, {
    let one = v / v;
    one / (one + (-v).exp())
});
unary_op!(Tanh, "tanh", v, v.tanh());
impl UnaryOpT for Gelu {
    const NAME: &'static str = "gelu";

    #[inline]
    fn bool(_: bool) -> bool {
        todo!("no unary function for bool")
    }
    #[inline]
    fn f8e4m3(v: F8E4M3) -> F8E4M3 {
        let half = F8E4M3::from(0.5f32);
        let one = F8E4M3::from(1.0f32);
        let sqrt_2_over_pi = F8E4M3::from(0.797_884_6_f32);
        let coeff = F8E4M3::from(0.044715f32);
        let x_cubed_term = v * v * v * coeff;
        v * half * (one + (sqrt_2_over_pi * (v + x_cubed_term)).tanh())
    }
    #[inline]
    fn f8e5m2(v: F8E5M2) -> F8E5M2 {
        let half = F8E5M2::from(0.5f32);
        let one = F8E5M2::from(1.0f32);
        let sqrt_2_over_pi = F8E5M2::from(0.797_884_6_f32);
        let coeff = F8E5M2::from(0.044715f32);
        let x_cubed_term = v * v * v * coeff;
        v * half * (one + (sqrt_2_over_pi * (v + x_cubed_term)).tanh())
    }
    #[inline]
    fn bf16(v: bf16) -> bf16 {
        let half = bf16::from_f32(0.5);
        let one = bf16::from_f32(1.0);
        let sqrt_2_over_pi = bf16::from_f32(0.797_884_6);
        let coeff = bf16::from_f32(0.044715);
        let x_cubed_term = v * v * v * coeff;
        v * half * (one + (sqrt_2_over_pi * (v + x_cubed_term)).tanh())
    }
    #[inline]
    fn f16(v: f16) -> f16 {
        let half = f16::from_f32(0.5);
        let one = f16::from_f32(1.0);
        let sqrt_2_over_pi = f16::from_f32(0.797_884_6);
        let coeff = f16::from_f32(0.044715);
        let x_cubed_term = v * v * v * coeff;
        v * half * (one + (sqrt_2_over_pi * (v + x_cubed_term)).tanh())
    }
    #[inline]
    fn f32(v: f32) -> f32 {
        let half = 0.5f32;
        let one = 1.0f32;
        let sqrt_2_over_pi = 0.797_884_6_f32;
        let coeff = 0.044715f32;
        let x_cubed_term = v * v * v * coeff;
        v * half * (one + (sqrt_2_over_pi * (v + x_cubed_term)).tanh())
    }
    #[inline]
    fn f64(v: f64) -> f64 {
        let half = 0.5f64;
        let one = 1.0f64;
        let sqrt_2_over_pi = 0.7978845608028654f64;
        let coeff = 0.044715f64;
        let x_cubed_term = v * v * v * coeff;
        v * half * (one + (sqrt_2_over_pi * (v + x_cubed_term)).tanh())
    }
    #[inline]
    fn u8(_: u8) -> u8 {
        todo!("no unary function for u8")
    }
    #[inline]
    fn u16(_: u16) -> u16 {
        todo!("no unary function for u16")
    }
    #[inline]
    fn u32(_: u32) -> u32 {
        todo!("no unary function for u32")
    }
    #[inline]
    fn u64(_: u64) -> u64 {
        todo!("no unary function for u64")
    }
    #[inline]
    fn i8(_: i8) -> i8 {
        todo!("no unary function for i8")
    }
    #[inline]
    fn i16(_: i16) -> i16 {
        todo!("no unary function for i16")
    }
    #[inline]
    fn i32(_: i32) -> i32 {
        todo!("no unary function for i32")
    }
    #[inline]
    fn i64(_: i64) -> i64 {
        todo!("no unary function for i64")
    }
}

unary_op!(Softplus, "softplus", v, (v.exp() + v / v).ln());
impl UnaryOpT for Silu {
    const NAME: &'static str = "silu";

    #[inline]
    fn bool(_: bool) -> bool {
        todo!("no unary function for bool")
    }
    #[inline]
    fn f8e4m3(v: F8E4M3) -> F8E4M3 {
        let one = F8E4M3::from(1.0f32);
        v * (one / (one + (-v).exp()))
    }
    #[inline]
    fn f8e5m2(v: F8E5M2) -> F8E5M2 {
        let one = F8E5M2::from(1.0f32);
        v * (one / (one + (-v).exp()))
    }
    #[inline]
    fn bf16(v: bf16) -> bf16 {
        let one = bf16::from_f32(1.0);
        v * (one / (one + (-v).exp()))
    }
    #[inline]
    fn f16(v: f16) -> f16 {
        let one = f16::from_f32(1.0);
        v * (one / (one + (-v).exp()))
    }
    #[inline]
    fn f32(v: f32) -> f32 {
        v * (1.0 / (1.0 + (-v).exp()))
    }
    #[inline]
    fn f64(v: f64) -> f64 {
        v * (1.0 / (1.0 + (-v).exp()))
    }
    #[inline]
    fn u8(_: u8) -> u8 {
        todo!("no unary function for u8")
    }
    #[inline]
    fn u16(_: u16) -> u16 {
        todo!("no unary function for u16")
    }
    #[inline]
    fn u32(_: u32) -> u32 {
        todo!("no unary function for u32")
    }
    #[inline]
    fn u64(_: u64) -> u64 {
        todo!("no unary function for u64")
    }
    #[inline]
    fn i8(_: i8) -> i8 {
        todo!("no unary function for i8")
    }
    #[inline]
    fn i16(_: i16) -> i16 {
        todo!("no unary function for i16")
    }
    #[inline]
    fn i32(_: i32) -> i32 {
        todo!("no unary function for i32")
    }
    #[inline]
    fn i64(_: i64) -> i64 {
        todo!("no unary function for i64")
    }
}
impl UnaryOpT for Mish {
    const NAME: &'static str = "mish";

    #[inline]
    fn bool(_: bool) -> bool {
        todo!("no unary function for bool")
    }
    #[inline]
    fn f8e4m3(v: F8E4M3) -> F8E4M3 {
        // mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        let exp_v = v.exp();
        let one = F8E4M3::from(1.0f32);
        let softplus = (one + exp_v).ln();
        v * softplus.tanh()
    }
    #[inline]
    fn f8e5m2(v: F8E5M2) -> F8E5M2 {
        let exp_v = v.exp();
        let one = F8E5M2::from(1.0f32);
        let softplus = (one + exp_v).ln();
        v * softplus.tanh()
    }
    #[inline]
    fn bf16(v: bf16) -> bf16 {
        let exp_v = v.exp();
        let one = bf16::from_f32(1.0);
        let softplus = (one + exp_v).ln();
        v * softplus.tanh()
    }
    #[inline]
    fn f16(v: f16) -> f16 {
        let exp_v = v.exp();
        let one = f16::from_f32(1.0);
        let softplus = (one + exp_v).ln();
        v * softplus.tanh()
    }
    #[inline]
    fn f32(v: f32) -> f32 {
        let exp_v = v.exp();
        let softplus = (1.0 + exp_v).ln();
        v * softplus.tanh()
    }
    #[inline]
    fn f64(v: f64) -> f64 {
        let exp_v = v.exp();
        let softplus = (1.0 + exp_v).ln();
        v * softplus.tanh()
    }
    #[inline]
    fn u8(_: u8) -> u8 {
        todo!("no unary function for u8")
    }
    #[inline]
    fn u16(_: u16) -> u16 {
        todo!("no unary function for u16")
    }
    #[inline]
    fn u32(_: u32) -> u32 {
        todo!("no unary function for u32")
    }
    #[inline]
    fn u64(_: u64) -> u64 {
        todo!("no unary function for u64")
    }
    #[inline]
    fn i8(_: i8) -> i8 {
        todo!("no unary function for i8")
    }
    #[inline]
    fn i16(_: i16) -> i16 {
        todo!("no unary function for i16")
    }
    #[inline]
    fn i32(_: i32) -> i32 {
        todo!("no unary function for i32")
    }
    #[inline]
    fn i64(_: i64) -> i64 {
        todo!("no unary function for i64")
    }
}

unary_op!(Sin, "sin", v, v.sin());
unary_op!(Cos, "cos", v, v.cos());
unary_op!(Tan, "tan", v, v.tan());
unary_op!(Exp, "exp", v, v.exp());
unary_op!(Exp2, "exp2", v, v.exp2());
impl UnaryOpT for Exp10 {
    const NAME: &'static str = "exp10";

    #[inline]
    fn bool(_: bool) -> bool {
        todo!("no unary function for bool")
    }
    #[inline]
    fn f8e4m3(v: F8E4M3) -> F8E4M3 {
        let ten = F8E4M3::from(10.0f32);
        ten.powf(v)
    }
    #[inline]
    fn f8e5m2(v: F8E5M2) -> F8E5M2 {
        let ten = F8E5M2::from(10.0f32);
        ten.powf(v)
    }
    #[inline]
    fn bf16(v: bf16) -> bf16 {
        let ten = bf16::from_f32(10.0);
        ten.powf(v)
    }
    #[inline]
    fn f16(v: f16) -> f16 {
        let ten = f16::from_f32(10.0);
        ten.powf(v)
    }
    #[inline]
    fn f32(v: f32) -> f32 {
        let ten = 10.0f32;
        ten.powf(v)
    }
    #[inline]
    fn f64(v: f64) -> f64 {
        let ten = 10.0f64;
        ten.powf(v)
    }
    #[inline]
    fn u8(_: u8) -> u8 {
        todo!("no unary function for u8")
    }
    #[inline]
    fn u16(_: u16) -> u16 {
        todo!("no unary function for u16")
    }
    #[inline]
    fn u32(_: u32) -> u32 {
        todo!("no unary function for u32")
    }
    #[inline]
    fn u64(_: u64) -> u64 {
        todo!("no unary function for u64")
    }
    #[inline]
    fn i8(_: i8) -> i8 {
        todo!("no unary function for i8")
    }
    #[inline]
    fn i16(_: i16) -> i16 {
        todo!("no unary function for i16")
    }
    #[inline]
    fn i32(_: i32) -> i32 {
        todo!("no unary function for i32")
    }
    #[inline]
    fn i64(_: i64) -> i64 {
        todo!("no unary function for i64")
    }
}
unary_op!(Ln, "ln", v, v.ln());
unary_op!(Log2, "log2", v, v.log2());
unary_op!(Log10, "log10", v, v.log10());

pub(crate) struct LogicalNot;

impl UnaryLogicalOpT for LogicalNot {
    const NAME: &'static str = "logical_not";

    #[inline]
    fn bool(v1: bool) -> bool {
        !v1
    }
    #[inline]
    fn f8e4m3(v1: F8E4M3) -> bool {
        v1 == F8E4M3::ZERO
    }
    #[inline]
    fn f8e5m2(v1: F8E5M2) -> bool {
        v1 == F8E5M2::ZERO
    }
    #[inline]
    fn bf16(v1: bf16) -> bool {
        v1 == bf16::ZERO
    }
    #[inline]
    fn f16(v1: f16) -> bool {
        v1 == f16::ZERO
    }
    #[inline]
    fn f32(v1: f32) -> bool {
        v1 == 0.0
    }
    #[inline]
    fn f64(v1: f64) -> bool {
        v1 == 0.0
    }
    #[inline]
    fn u8(v1: u8) -> bool {
        v1 == 0
    }
    #[inline]
    fn u16(v1: u16) -> bool {
        v1 == 0
    }
    #[inline]
    fn u32(v1: u32) -> bool {
        v1 == 0
    }
    #[inline]
    fn u64(v1: u64) -> bool {
        v1 == 0
    }
    #[inline]
    fn i8(v1: i8) -> bool {
        v1 == 0
    }
    #[inline]
    fn i16(v1: i16) -> bool {
        v1 == 0
    }
    #[inline]
    fn i32(v1: i32) -> bool {
        v1 == 0
    }
    #[inline]
    fn i64(v1: i64) -> bool {
        v1 == 0
    }
}

pub(crate) struct AddScalar;
pub(crate) struct SubScalar;
pub(crate) struct MulScalar;
pub(crate) struct DivScalar;
pub(crate) struct PowScalar;
pub(crate) struct MaximumScalar;
pub(crate) struct MinimumScalar;
pub(crate) struct LeakyRelu;
pub(crate) struct Elu;
pub(crate) struct Prelu;

macro_rules! unary_scalar_op {
    ($op:ident, $name: literal, $e: expr) => {
        impl UnaryScalarOpT for $op {
            const NAME: &'static str = $name;

            #[inline]
            fn bool(_: bool, _: Scalar) -> bool {
                todo!("no unary scalar function for bool")
            }
            #[inline]
            fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> F8E4M3 {
                $e(v1, scalar.to_f8e4m3())
            }
            #[inline]
            fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> F8E5M2 {
                $e(v1, scalar.to_f8e5m2())
            }
            #[inline]
            fn bf16(v1: bf16, scalar: Scalar) -> bf16 {
                $e(v1, scalar.to_bf16())
            }
            #[inline]
            fn f16(v1: f16, scalar: Scalar) -> f16 {
                $e(v1, scalar.to_f16())
            }
            #[inline]
            fn f32(v1: f32, scalar: Scalar) -> f32 {
                $e(v1, scalar.to_f32())
            }
            #[inline]
            fn f64(v1: f64, scalar: Scalar) -> f64 {
                $e(v1, scalar.to_f64())
            }
            #[inline]
            fn u8(_: u8, _: Scalar) -> u8 {
                todo!("no unary scalar function for u8")
            }
            #[inline]
            fn u16(_: u16, _: Scalar) -> u16 {
                todo!("no unary scalar function for u16")
            }
            #[inline]
            fn u32(_: u32, _: Scalar) -> u32 {
                todo!("no unary scalar function for u32")
            }
            #[inline]
            fn u64(_: u64, _: Scalar) -> u64 {
                todo!("no unary scalar function for u64")
            }
            #[inline]
            fn i8(_: i8, _: Scalar) -> i8 {
                todo!("no unary scalar function for i8")
            }
            #[inline]
            fn i16(_: i16, _: Scalar) -> i16 {
                todo!("no unary scalar function for i16")
            }
            #[inline]
            fn i32(_: i32, _: Scalar) -> i32 {
                todo!("no unary scalar function for i32")
            }
            #[inline]
            fn i64(_: i64, _: Scalar) -> i64 {
                todo!("no unary scalar function for i64")
            }
        }
    };
}

impl UnaryScalarOpT for AddScalar {
    const NAME: &'static str = "add_scalar";

    #[inline]
    fn bool(_: bool, _: Scalar) -> bool {
        todo!("no unary scalar function for bool")
    }
    #[inline]
    fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> F8E4M3 {
        v1 + scalar.to_f8e4m3()
    }
    #[inline]
    fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> F8E5M2 {
        v1 + scalar.to_f8e5m2()
    }
    #[inline]
    fn bf16(v1: bf16, scalar: Scalar) -> bf16 {
        v1 + scalar.to_bf16()
    }
    #[inline]
    fn f16(v1: f16, scalar: Scalar) -> f16 {
        v1 + scalar.to_f16()
    }
    #[inline]
    fn f32(v1: f32, scalar: Scalar) -> f32 {
        v1 + scalar.to_f32()
    }
    #[inline]
    fn f64(v1: f64, scalar: Scalar) -> f64 {
        v1 + scalar.to_f64()
    }
    #[inline]
    fn u8(v1: u8, scalar: Scalar) -> u8 {
        v1 + scalar.to_u8()
    }
    #[inline]
    fn u16(v1: u16, scalar: Scalar) -> u16 {
        v1 + scalar.to_u16()
    }
    #[inline]
    fn u32(v1: u32, scalar: Scalar) -> u32 {
        v1 + scalar.to_u32()
    }
    #[inline]
    fn u64(v1: u64, scalar: Scalar) -> u64 {
        v1 + scalar.to_u64()
    }
    #[inline]
    fn i8(v1: i8, scalar: Scalar) -> i8 {
        v1 + scalar.to_i8()
    }
    #[inline]
    fn i16(v1: i16, scalar: Scalar) -> i16 {
        v1 + scalar.to_i16()
    }
    #[inline]
    fn i32(v1: i32, scalar: Scalar) -> i32 {
        v1 + scalar.to_i32()
    }
    #[inline]
    fn i64(v1: i64, scalar: Scalar) -> i64 {
        v1 + scalar.to_i64()
    }
}
impl UnaryScalarOpT for SubScalar {
    const NAME: &'static str = "sub_scalar";

    #[inline]
    fn bool(_: bool, _: Scalar) -> bool {
        todo!("no unary scalar function for bool")
    }
    #[inline]
    fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> F8E4M3 {
        v1 - scalar.to_f8e4m3()
    }
    #[inline]
    fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> F8E5M2 {
        v1 - scalar.to_f8e5m2()
    }
    #[inline]
    fn bf16(v1: bf16, scalar: Scalar) -> bf16 {
        v1 - scalar.to_bf16()
    }
    #[inline]
    fn f16(v1: f16, scalar: Scalar) -> f16 {
        v1 - scalar.to_f16()
    }
    #[inline]
    fn f32(v1: f32, scalar: Scalar) -> f32 {
        v1 - scalar.to_f32()
    }
    #[inline]
    fn f64(v1: f64, scalar: Scalar) -> f64 {
        v1 - scalar.to_f64()
    }
    #[inline]
    fn u8(v1: u8, scalar: Scalar) -> u8 {
        v1 - scalar.to_u8()
    }
    #[inline]
    fn u16(v1: u16, scalar: Scalar) -> u16 {
        v1 - scalar.to_u16()
    }
    #[inline]
    fn u32(v1: u32, scalar: Scalar) -> u32 {
        v1 - scalar.to_u32()
    }
    #[inline]
    fn u64(v1: u64, scalar: Scalar) -> u64 {
        v1 - scalar.to_u64()
    }
    #[inline]
    fn i8(v1: i8, scalar: Scalar) -> i8 {
        v1 - scalar.to_i8()
    }
    #[inline]
    fn i16(v1: i16, scalar: Scalar) -> i16 {
        v1 - scalar.to_i16()
    }
    #[inline]
    fn i32(v1: i32, scalar: Scalar) -> i32 {
        v1 - scalar.to_i32()
    }
    #[inline]
    fn i64(v1: i64, scalar: Scalar) -> i64 {
        v1 - scalar.to_i64()
    }
}
impl UnaryScalarOpT for MulScalar {
    const NAME: &'static str = "mul_scalar";

    #[inline]
    fn bool(_: bool, _: Scalar) -> bool {
        todo!("no unary scalar function for bool")
    }
    #[inline]
    fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> F8E4M3 {
        v1 * scalar.to_f8e4m3()
    }
    #[inline]
    fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> F8E5M2 {
        v1 * scalar.to_f8e5m2()
    }
    #[inline]
    fn bf16(v1: bf16, scalar: Scalar) -> bf16 {
        v1 * scalar.to_bf16()
    }
    #[inline]
    fn f16(v1: f16, scalar: Scalar) -> f16 {
        v1 * scalar.to_f16()
    }
    #[inline]
    fn f32(v1: f32, scalar: Scalar) -> f32 {
        v1 * scalar.to_f32()
    }
    #[inline]
    fn f64(v1: f64, scalar: Scalar) -> f64 {
        v1 * scalar.to_f64()
    }
    #[inline]
    fn u8(v1: u8, scalar: Scalar) -> u8 {
        v1 * scalar.to_u8()
    }
    #[inline]
    fn u16(v1: u16, scalar: Scalar) -> u16 {
        v1 * scalar.to_u16()
    }
    #[inline]
    fn u32(v1: u32, scalar: Scalar) -> u32 {
        v1 * scalar.to_u32()
    }
    #[inline]
    fn u64(v1: u64, scalar: Scalar) -> u64 {
        v1 * scalar.to_u64()
    }
    #[inline]
    fn i8(v1: i8, scalar: Scalar) -> i8 {
        v1 * scalar.to_i8()
    }
    #[inline]
    fn i16(v1: i16, scalar: Scalar) -> i16 {
        v1 * scalar.to_i16()
    }
    #[inline]
    fn i32(v1: i32, scalar: Scalar) -> i32 {
        v1 * scalar.to_i32()
    }
    #[inline]
    fn i64(v1: i64, scalar: Scalar) -> i64 {
        v1 * scalar.to_i64()
    }
}
unary_scalar_op!(DivScalar, "div_scalar", |v1, v2| v1 / v2);
impl UnaryScalarOpT for PowScalar {
    const NAME: &'static str = "pow_scalar";

    #[inline]
    fn bool(_: bool, _: Scalar) -> bool {
        todo!("no unary scalar function for bool")
    }
    #[inline]
    fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> F8E4M3 {
        v1.powf(scalar.to_f8e4m3())
    }
    #[inline]
    fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> F8E5M2 {
        v1.powf(scalar.to_f8e5m2())
    }
    #[inline]
    fn bf16(v1: bf16, scalar: Scalar) -> bf16 {
        v1.powf(scalar.to_bf16())
    }
    #[inline]
    fn f16(v1: f16, scalar: Scalar) -> f16 {
        v1.powf(scalar.to_f16())
    }
    #[inline]
    fn f32(v1: f32, scalar: Scalar) -> f32 {
        v1.powf(scalar.to_f32())
    }
    #[inline]
    fn f64(v1: f64, scalar: Scalar) -> f64 {
        v1.powf(scalar.to_f64())
    }
    #[inline]
    fn u8(v1: u8, scalar: Scalar) -> u8 {
        v1.pow(scalar.to_u32())
    }
    #[inline]
    fn u16(v1: u16, scalar: Scalar) -> u16 {
        v1.pow(scalar.to_u32())
    }
    #[inline]
    fn u32(v1: u32, scalar: Scalar) -> u32 {
        v1.pow(scalar.to_u32())
    }
    #[inline]
    fn u64(v1: u64, scalar: Scalar) -> u64 {
        v1.pow(scalar.to_u32())
    }
    #[inline]
    fn i8(v1: i8, scalar: Scalar) -> i8 {
        v1.pow(scalar.to_u32())
    }
    #[inline]
    fn i16(v1: i16, scalar: Scalar) -> i16 {
        v1.pow(scalar.to_u32())
    }
    #[inline]
    fn i32(v1: i32, scalar: Scalar) -> i32 {
        v1.pow(scalar.to_u32())
    }
    #[inline]
    fn i64(v1: i64, scalar: Scalar) -> i64 {
        v1.pow(scalar.to_u32())
    }
}
impl UnaryScalarOpT for MaximumScalar {
    const NAME: &'static str = "maximum_scalar";

    #[inline]
    fn bool(_: bool, _: Scalar) -> bool {
        todo!("no unary scalar function for bool")
    }
    #[inline]
    fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> F8E4M3 {
        let v2 = scalar.to_f8e4m3();
        if v1 < v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> F8E5M2 {
        let v2 = scalar.to_f8e5m2();
        if v1 < v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn bf16(v1: bf16, scalar: Scalar) -> bf16 {
        let v2 = scalar.to_bf16();
        if v1 < v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn f16(v1: f16, scalar: Scalar) -> f16 {
        let v2 = scalar.to_f16();
        if v1 < v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn f32(v1: f32, scalar: Scalar) -> f32 {
        let v2 = scalar.to_f32();
        if v1 < v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn f64(v1: f64, scalar: Scalar) -> f64 {
        let v2 = scalar.to_f64();
        if v1 < v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn u8(v1: u8, scalar: Scalar) -> u8 {
        let v2 = scalar.to_u8();
        if v1 < v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn u16(v1: u16, scalar: Scalar) -> u16 {
        let v2 = scalar.to_u16();
        if v1 < v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn u32(v1: u32, scalar: Scalar) -> u32 {
        let v2 = scalar.to_u32();
        if v1 < v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn u64(v1: u64, scalar: Scalar) -> u64 {
        let v2 = scalar.to_u64();
        if v1 < v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn i8(v1: i8, scalar: Scalar) -> i8 {
        let v2 = scalar.to_i8();
        if v1 < v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn i16(v1: i16, scalar: Scalar) -> i16 {
        let v2 = scalar.to_i16();
        if v1 < v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn i32(v1: i32, scalar: Scalar) -> i32 {
        let v2 = scalar.to_i32();
        if v1 < v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn i64(v1: i64, scalar: Scalar) -> i64 {
        let v2 = scalar.to_i64();
        if v1 < v2 {
            v2
        } else {
            v1
        }
    }
}
impl UnaryScalarOpT for MinimumScalar {
    const NAME: &'static str = "minimum_scalar";

    #[inline]
    fn bool(_: bool, _: Scalar) -> bool {
        todo!("no unary scalar function for bool")
    }
    #[inline]
    fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> F8E4M3 {
        let v2 = scalar.to_f8e4m3();
        if v1 > v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> F8E5M2 {
        let v2 = scalar.to_f8e5m2();
        if v1 > v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn bf16(v1: bf16, scalar: Scalar) -> bf16 {
        let v2 = scalar.to_bf16();
        if v1 > v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn f16(v1: f16, scalar: Scalar) -> f16 {
        let v2 = scalar.to_f16();
        if v1 > v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn f32(v1: f32, scalar: Scalar) -> f32 {
        let v2 = scalar.to_f32();
        if v1 > v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn f64(v1: f64, scalar: Scalar) -> f64 {
        let v2 = scalar.to_f64();
        if v1 > v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn u8(v1: u8, scalar: Scalar) -> u8 {
        let v2 = scalar.to_u8();
        if v1 > v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn u16(v1: u16, scalar: Scalar) -> u16 {
        let v2 = scalar.to_u16();
        if v1 > v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn u32(v1: u32, scalar: Scalar) -> u32 {
        let v2 = scalar.to_u32();
        if v1 > v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn u64(v1: u64, scalar: Scalar) -> u64 {
        let v2 = scalar.to_u64();
        if v1 > v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn i8(v1: i8, scalar: Scalar) -> i8 {
        let v2 = scalar.to_i8();
        if v1 > v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn i16(v1: i16, scalar: Scalar) -> i16 {
        let v2 = scalar.to_i16();
        if v1 > v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn i32(v1: i32, scalar: Scalar) -> i32 {
        let v2 = scalar.to_i32();
        if v1 > v2 {
            v2
        } else {
            v1
        }
    }
    #[inline]
    fn i64(v1: i64, scalar: Scalar) -> i64 {
        let v2 = scalar.to_i64();
        if v1 > v2 {
            v2
        } else {
            v1
        }
    }
}
impl UnaryScalarOpT for LeakyRelu {
    const NAME: &'static str = "leaky_relu";

    #[inline]
    fn bool(_: bool, _: Scalar) -> bool {
        todo!("no unary scalar function for bool")
    }
    #[inline]
    fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> F8E4M3 {
        let alpha = scalar.to_f8e4m3();
        if v1 > F8E4M3::ZERO {
            v1
        } else {
            v1 * alpha
        }
    }
    #[inline]
    fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> F8E5M2 {
        let alpha = scalar.to_f8e5m2();
        if v1 > F8E5M2::ZERO {
            v1
        } else {
            v1 * alpha
        }
    }
    #[inline]
    fn bf16(v1: bf16, scalar: Scalar) -> bf16 {
        let alpha = scalar.to_bf16();
        if v1 > bf16::ZERO {
            v1
        } else {
            v1 * alpha
        }
    }
    #[inline]
    fn f16(v1: f16, scalar: Scalar) -> f16 {
        let alpha = scalar.to_f16();
        if v1 > f16::ZERO {
            v1
        } else {
            v1 * alpha
        }
    }
    #[inline]
    fn f32(v1: f32, scalar: Scalar) -> f32 {
        let alpha = scalar.to_f32();
        if v1 > 0.0 {
            v1
        } else {
            v1 * alpha
        }
    }
    #[inline]
    fn f64(v1: f64, scalar: Scalar) -> f64 {
        let alpha = scalar.to_f64();
        if v1 > 0.0 {
            v1
        } else {
            v1 * alpha
        }
    }
    #[inline]
    fn u8(_: u8, _: Scalar) -> u8 {
        todo!("no unary scalar function for u8")
    }
    #[inline]
    fn u16(_: u16, _: Scalar) -> u16 {
        todo!("no unary scalar function for u16")
    }
    #[inline]
    fn u32(_: u32, _: Scalar) -> u32 {
        todo!("no unary scalar function for u32")
    }
    #[inline]
    fn u64(_: u64, _: Scalar) -> u64 {
        todo!("no unary scalar function for u64")
    }
    #[inline]
    fn i8(_: i8, _: Scalar) -> i8 {
        todo!("no unary scalar function for i8")
    }
    #[inline]
    fn i16(_: i16, _: Scalar) -> i16 {
        todo!("no unary scalar function for i16")
    }
    #[inline]
    fn i32(_: i32, _: Scalar) -> i32 {
        todo!("no unary scalar function for i32")
    }
    #[inline]
    fn i64(_: i64, _: Scalar) -> i64 {
        todo!("no unary scalar function for i64")
    }
}
impl UnaryScalarOpT for Elu {
    const NAME: &'static str = "elu";

    #[inline]
    fn bool(_: bool, _: Scalar) -> bool {
        todo!("no unary scalar function for bool")
    }
    #[inline]
    fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> F8E4M3 {
        let alpha = scalar.to_f8e4m3();
        if v1 > F8E4M3::ZERO {
            v1
        } else {
            alpha * (v1.exp() - F8E4M3::ONE)
        }
    }
    #[inline]
    fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> F8E5M2 {
        let alpha = scalar.to_f8e5m2();
        if v1 > F8E5M2::ZERO {
            v1
        } else {
            alpha * (v1.exp() - F8E5M2::ONE)
        }
    }
    #[inline]
    fn bf16(v1: bf16, scalar: Scalar) -> bf16 {
        let alpha = scalar.to_bf16();
        if v1 > bf16::ZERO {
            v1
        } else {
            alpha * (v1.exp() - bf16::ONE)
        }
    }
    #[inline]
    fn f16(v1: f16, scalar: Scalar) -> f16 {
        let alpha = scalar.to_f16();
        if v1 > f16::ZERO {
            v1
        } else {
            alpha * (v1.exp() - f16::ONE)
        }
    }
    #[inline]
    fn f32(v1: f32, scalar: Scalar) -> f32 {
        let alpha = scalar.to_f32();
        if v1 > 0.0 {
            v1
        } else {
            alpha * (v1.exp() - 1.0)
        }
    }
    #[inline]
    fn f64(v1: f64, scalar: Scalar) -> f64 {
        let alpha = scalar.to_f64();
        if v1 > 0.0 {
            v1
        } else {
            alpha * (v1.exp() - 1.0)
        }
    }
    #[inline]
    fn u8(_: u8, _: Scalar) -> u8 {
        todo!("no unary scalar function for u8")
    }
    #[inline]
    fn u16(_: u16, _: Scalar) -> u16 {
        todo!("no unary scalar function for u16")
    }
    #[inline]
    fn u32(_: u32, _: Scalar) -> u32 {
        todo!("no unary scalar function for u32")
    }
    #[inline]
    fn u64(_: u64, _: Scalar) -> u64 {
        todo!("no unary scalar function for u64")
    }
    #[inline]
    fn i8(_: i8, _: Scalar) -> i8 {
        todo!("no unary scalar function for i8")
    }
    #[inline]
    fn i16(_: i16, _: Scalar) -> i16 {
        todo!("no unary scalar function for i16")
    }
    #[inline]
    fn i32(_: i32, _: Scalar) -> i32 {
        todo!("no unary scalar function for i32")
    }
    #[inline]
    fn i64(_: i64, _: Scalar) -> i64 {
        todo!("no unary scalar function for i64")
    }
}

impl UnaryScalarOpT for Prelu {
    const NAME: &'static str = "prelu";

    #[inline]
    fn bool(_: bool, _: Scalar) -> bool {
        todo!("no unary scalar function for bool")
    }
    #[inline]
    fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> F8E4M3 {
        let alpha = scalar.to_f8e4m3();
        if v1 > F8E4M3::ZERO {
            v1
        } else {
            alpha * v1
        }
    }
    #[inline]
    fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> F8E5M2 {
        let alpha = scalar.to_f8e5m2();
        if v1 > F8E5M2::ZERO {
            v1
        } else {
            alpha * v1
        }
    }
    #[inline]
    fn bf16(v1: bf16, scalar: Scalar) -> bf16 {
        let alpha = scalar.to_bf16();
        if v1 > bf16::ZERO {
            v1
        } else {
            alpha * v1
        }
    }
    #[inline]
    fn f16(v1: f16, scalar: Scalar) -> f16 {
        let alpha = scalar.to_f16();
        if v1 > f16::ZERO {
            v1
        } else {
            alpha * v1
        }
    }
    #[inline]
    fn f32(v1: f32, scalar: Scalar) -> f32 {
        let alpha = scalar.to_f32();
        if v1 > 0.0 {
            v1
        } else {
            alpha * v1
        }
    }
    #[inline]
    fn f64(v1: f64, scalar: Scalar) -> f64 {
        let alpha = scalar.to_f64();
        if v1 > 0.0 {
            v1
        } else {
            alpha * v1
        }
    }
    #[inline]
    fn u8(_: u8, _: Scalar) -> u8 {
        todo!("no unary scalar function for u8")
    }
    #[inline]
    fn u16(_: u16, _: Scalar) -> u16 {
        todo!("no unary scalar function for u16")
    }
    #[inline]
    fn u32(_: u32, _: Scalar) -> u32 {
        todo!("no unary scalar function for u32")
    }
    #[inline]
    fn u64(_: u64, _: Scalar) -> u64 {
        todo!("no unary scalar function for u64")
    }
    #[inline]
    fn i8(_: i8, _: Scalar) -> i8 {
        todo!("no unary scalar function for i8")
    }
    #[inline]
    fn i16(_: i16, _: Scalar) -> i16 {
        todo!("no unary scalar function for i16")
    }
    #[inline]
    fn i32(_: i32, _: Scalar) -> i32 {
        todo!("no unary scalar function for i32")
    }
    #[inline]
    fn i64(_: i64, _: Scalar) -> i64 {
        todo!("no unary scalar function for i64")
    }
}
