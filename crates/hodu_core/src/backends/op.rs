#![allow(clippy::bool_comparison, clippy::eq_op)]

// Naming convention for kernel names to avoid duplication:
// BinaryLogicalOp -> binary_logical_and (not binary_logical_logical_and)
// UnaryLogicalOp -> unary_logical_not (not unary_logical_logical_not)

pub mod conv;

use crate::{compat::*, scalar::Scalar, tensor::TensorId};

use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};
use num_traits::float::Float;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Maximum, // no-backprop
    Minimum, // no-backprop
}

pub trait BinaryOpT {
    const NAME: &'static str;
    const KERNEL: &'static str;

    fn bool(v1: bool, v2: bool) -> bool;
    fn f8e4m3(v1: F8E4M3, v2: F8E4M3) -> F8E4M3;
    fn f8e5m2(v1: F8E5M2, v2: F8E5M2) -> F8E5M2;
    fn bf16(v1: bf16, v2: bf16) -> bf16;
    fn f16(v1: f16, v2: f16) -> f16;
    fn f32(v1: f32, v2: f32) -> f32;
    fn f64(v1: f64, v2: f64) -> f64;
    fn u8(v1: u8, v2: u8) -> u8;
    fn u16(v1: u16, v2: u16) -> u16;
    fn u32(v1: u32, v2: u32) -> u32;
    fn u64(v1: u64, v2: u64) -> u64;
    fn i8(v1: i8, v2: i8) -> i8;
    fn i16(v1: i16, v2: i16) -> i16;
    fn i32(v1: i32, v2: i32) -> i32;
    fn i64(v1: i64, v2: i64) -> i64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum BinaryLogicalOp {
    LogicalAnd, // no-backprop
    LogicalOr,  // no-backprop
    LogicalXor, // no-backprop
}

pub trait BinaryLogicalOpT {
    const NAME: &'static str;
    const KERNEL: &'static str;

    fn bool(v1: bool, v2: bool) -> bool;
    fn f8e4m3(v1: F8E4M3, v2: F8E4M3) -> bool;
    fn f8e5m2(v1: F8E5M2, v2: F8E5M2) -> bool;
    fn bf16(v1: bf16, v2: bf16) -> bool;
    fn f16(v1: f16, v2: f16) -> bool;
    fn f32(v1: f32, v2: f32) -> bool;
    fn f64(v1: f64, v2: f64) -> bool;
    fn u8(v1: u8, v2: u8) -> bool;
    fn u16(v1: u16, v2: u16) -> bool;
    fn u32(v1: u32, v2: u32) -> bool;
    fn u64(v1: u64, v2: u64) -> bool;
    fn i8(v1: i8, v2: i8) -> bool;
    fn i16(v1: i16, v2: i16) -> bool;
    fn i32(v1: i32, v2: i32) -> bool;
    fn i64(v1: i64, v2: i64) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum CmpOp {
    Eq, // no-backprop
    Ne, // no-backprop
    Lt, // no-backprop
    Le, // no-backprop
    Gt, // no-backprop
    Ge, // no-backprop
}

pub trait CmpOpT {
    const NAME: &'static str;
    const KERNEL: &'static str;

    fn bool(v1: bool, v2: bool) -> bool;
    fn f8e4m3(v1: F8E4M3, v2: F8E4M3) -> bool;
    fn f8e5m2(v1: F8E5M2, v2: F8E5M2) -> bool;
    fn bf16(v1: bf16, v2: bf16) -> bool;
    fn f16(v1: f16, v2: f16) -> bool;
    fn f32(v1: f32, v2: f32) -> bool;
    fn f64(v1: f64, v2: f64) -> bool;
    fn u8(v1: u8, v2: u8) -> bool;
    fn u16(v1: u16, v2: u16) -> bool;
    fn u32(v1: u32, v2: u32) -> bool;
    fn u64(v1: u64, v2: u64) -> bool;
    fn i8(v1: i8, v2: i8) -> bool;
    fn i16(v1: i16, v2: i16) -> bool;
    fn i32(v1: i32, v2: i32) -> bool;
    fn i64(v1: i64, v2: i64) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum CmpScalarOp {
    EqScalar, // no-backprop
    NeScalar, // no-backprop
    LtScalar, // no-backprop
    LeScalar, // no-backprop
    GtScalar, // no-backprop
    GeScalar, // no-backprop
}

pub trait CmpScalarOpT {
    const NAME: &'static str;
    const KERNEL: &'static str;

    fn bool(v1: bool, scalar: Scalar) -> bool;
    fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> bool;
    fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> bool;
    fn bf16(v1: bf16, scalar: Scalar) -> bool;
    fn f16(v1: f16, scalar: Scalar) -> bool;
    fn f32(v1: f32, scalar: Scalar) -> bool;
    fn f64(v1: f64, scalar: Scalar) -> bool;
    fn u8(v1: u8, scalar: Scalar) -> bool;
    fn u16(v1: u16, scalar: Scalar) -> bool;
    fn u32(v1: u32, scalar: Scalar) -> bool;
    fn u64(v1: u64, scalar: Scalar) -> bool;
    fn i8(v1: i8, scalar: Scalar) -> bool;
    fn i16(v1: i16, scalar: Scalar) -> bool;
    fn i32(v1: i32, scalar: Scalar) -> bool;
    fn i64(v1: i64, scalar: Scalar) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum UnaryOp {
    Neg,
    Abs,  // no-backprop
    Sign, // no-backprop
    Square,
    Sqrt,
    Recip,

    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    Softplus,

    Sin,
    Cos,
    Tan,

    Exp,
    Exp2,
    Exp10,
    Ln,
    Log2,
    Log10,
}

pub trait UnaryOpT {
    const NAME: &'static str;
    const KERNEL: &'static str;

    fn bool(v1: bool) -> bool;
    fn f8e4m3(v1: F8E4M3) -> F8E4M3;
    fn f8e5m2(v1: F8E5M2) -> F8E5M2;
    fn bf16(v1: bf16) -> bf16;
    fn f16(v1: f16) -> f16;
    fn f32(v1: f32) -> f32;
    fn f64(v1: f64) -> f64;
    fn u8(v1: u8) -> u8;
    fn u16(v1: u16) -> u16;
    fn u32(v1: u32) -> u32;
    fn u64(v1: u64) -> u64;
    fn i8(v1: i8) -> i8;
    fn i16(v1: i16) -> i16;
    fn i32(v1: i32) -> i32;
    fn i64(v1: i64) -> i64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum UnaryLogicalOp {
    LogicalNot, // no-backprop
}

pub trait UnaryLogicalOpT {
    const NAME: &'static str;
    const KERNEL: &'static str;

    fn bool(v1: bool) -> bool;
    fn f8e4m3(v1: F8E4M3) -> bool;
    fn f8e5m2(v1: F8E5M2) -> bool;
    fn bf16(v1: bf16) -> bool;
    fn f16(v1: f16) -> bool;
    fn f32(v1: f32) -> bool;
    fn f64(v1: f64) -> bool;
    fn u8(v1: u8) -> bool;
    fn u16(v1: u16) -> bool;
    fn u32(v1: u32) -> bool;
    fn u64(v1: u64) -> bool;
    fn i8(v1: i8) -> bool;
    fn i16(v1: i16) -> bool;
    fn i32(v1: i32) -> bool;
    fn i64(v1: i64) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum UnaryScalarOp {
    AddScalar,
    SubScalar,
    MulScalar,
    DivScalar,
    PowScalar,
    MaximumScalar, // no-backprop
    MinimumScalar, // no-backprop

    LeakyRelu,
    Elu,
}

pub trait UnaryScalarOpT {
    const NAME: &'static str;
    const KERNEL: &'static str;

    fn bool(v1: bool, scalar: Scalar) -> bool;
    fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> F8E4M3;
    fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> F8E5M2;
    fn bf16(v1: bf16, scalar: Scalar) -> bf16;
    fn f16(v1: f16, scalar: Scalar) -> f16;
    fn f32(v1: f32, scalar: Scalar) -> f32;
    fn f64(v1: f64, scalar: Scalar) -> f64;
    fn u8(v1: u8, scalar: Scalar) -> u8;
    fn u16(v1: u16, scalar: Scalar) -> u16;
    fn u32(v1: u32, scalar: Scalar) -> u32;
    fn u64(v1: u64, scalar: Scalar) -> u64;
    fn i8(v1: i8, scalar: Scalar) -> i8;
    fn i16(v1: i16, scalar: Scalar) -> i16;
    fn i32(v1: i32, scalar: Scalar) -> i32;
    fn i64(v1: i64, scalar: Scalar) -> i64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum MatrixOp {
    Matmul,
    Dot, // Supports 1D dot product, 2D matmul, and ND broadcast matmul
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum ReduceOp {
    Sum,
    Mean,
    Max, // no-backprop
    Min, // no-backprop
    Prod,
    Std,
    Var,
    Norm,
    ArgMax, // no-backprop
    ArgMin, // no-backprop
}

impl ReduceOp {
    pub fn to_string(&self) -> String {
        match self {
            ReduceOp::Sum => "sum".to_string(),
            ReduceOp::Mean => "mean".to_string(),
            ReduceOp::Max => "max".to_string(),
            ReduceOp::Min => "min".to_string(),
            ReduceOp::Prod => "prod".to_string(),
            ReduceOp::Std => "std".to_string(),
            ReduceOp::Var => "var".to_string(),
            ReduceOp::Norm => "norm".to_string(),
            ReduceOp::ArgMax => "argmax".to_string(),
            ReduceOp::ArgMin => "argmin".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum ConcatOp {
    Concat,
    Stack,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum SplitOp {
    Split,
    Chunk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum IndexingOp {
    IndexSelect,
    Gather,
    Scatter,
    ScatterAdd,
    ScatterMax,
    ScatterMin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum ConvOp {
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    Conv1dGradWeight,
    Conv2dGradWeight,
    Conv3dGradWeight,
    ConvTranspose1dGradWeight,
    ConvTranspose2dGradWeight,
    ConvTranspose3dGradWeight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum ShapeOp {
    Reshape,
    Flatten,
    Squeeze,
    Unsqueeze,
    Broadcast,
    Transpose,
    Permute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum CastOp {
    ToDType, // no-backprop
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum MemoryOp {
    Contiguous, // no-backprop
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum Op {
    Binary(BinaryOp, TensorId, TensorId),
    BinaryLogical(BinaryLogicalOp, TensorId, TensorId),
    Cmp(CmpOp, TensorId, TensorId),
    CmpScalar(CmpScalarOp, TensorId, Scalar),
    Unary(UnaryOp, TensorId),
    UnaryLogical(UnaryLogicalOp, TensorId),
    UnaryScalar(UnaryScalarOp, TensorId, Scalar),
    Matrix(MatrixOp, TensorId, TensorId),
    Reduce(ReduceOp, TensorId, Vec<Scalar>),
    Concat(ConcatOp, Vec<TensorId>, Vec<Scalar>),
    Split(SplitOp, TensorId, Vec<Scalar>, usize),
    Indexing(IndexingOp, Vec<TensorId>, Vec<Scalar>),
    Conv(ConvOp, TensorId, TensorId, Vec<Scalar>),
    Shape(ShapeOp, TensorId),
    Cast(CastOp, TensorId),
    Memory(MemoryOp, TensorId),
    Dummy,
}

impl Op {
    pub fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        match self {
            Op::Binary(_, t1, t2) => vec![*t1, *t2],
            Op::BinaryLogical(_, t1, t2) => vec![*t1, *t2],
            Op::Cmp(_, t1, t2) => vec![*t1, *t2],
            Op::CmpScalar(_, t, _) => vec![*t],
            Op::Unary(_, t) => vec![*t],
            Op::UnaryLogical(_, t) => vec![*t],
            Op::UnaryScalar(_, t, _) => vec![*t],
            Op::Matrix(_, t1, t2) => vec![*t1, *t2],
            Op::Reduce(_, t, _) => vec![*t],
            Op::Concat(_, tt, _) => tt.clone(),
            Op::Split(_, t, _, _) => vec![*t],
            Op::Indexing(_, tt, _) => tt.clone(),
            Op::Conv(_, input, weight, _) => vec![*input, *weight],
            Op::Shape(_, t) => vec![*t],
            Op::Cast(_, t) => vec![*t],
            Op::Memory(_, t) => vec![*t],
            Op::Dummy => vec![],
        }
    }
}

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
            const KERNEL: &'static str = concat!("binary_", $name);

            #[inline(always)]
            fn bool(_: bool, _: bool) -> bool {
                todo!("no binary function for bool")
            }
            #[inline(always)]
            fn f8e4m3(v1: F8E4M3, v2: F8E4M3) -> F8E4M3 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f8e5m2(v1: F8E5M2, v2: F8E5M2) -> F8E5M2 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn bf16(v1: bf16, v2: bf16) -> bf16 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f16(v1: f16, v2: f16) -> f16 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f32(v1: f32, v2: f32) -> f32 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f64(v1: f64, v2: f64) -> f64 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u8(v1: u8, v2: u8) -> u8 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u16(v1: u16, v2: u16) -> u16 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u32(v1: u32, v2: u32) -> u32 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u64(v1: u64, v2: u64) -> u64 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn i8(v1: i8, v2: i8) -> i8 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn i16(v1: i16, v2: i16) -> i16 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn i32(v1: i32, v2: i32) -> i32 {
                $e(v1, v2)
            }
            #[inline(always)]
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
    const KERNEL: &'static str = "binary_pow";

    #[inline(always)]
    fn bool(_: bool, _: bool) -> bool {
        todo!("no binary function for bool")
    }
    #[inline(always)]
    fn f8e4m3(v1: F8E4M3, v2: F8E4M3) -> F8E4M3 {
        v1.powf(v2)
    }
    #[inline(always)]
    fn f8e5m2(v1: F8E5M2, v2: F8E5M2) -> F8E5M2 {
        v1.powf(v2)
    }
    #[inline(always)]
    fn bf16(v1: bf16, v2: bf16) -> bf16 {
        v1.powf(v2)
    }
    #[inline(always)]
    fn f16(v1: f16, v2: f16) -> f16 {
        v1.powf(v2)
    }
    #[inline(always)]
    fn f32(v1: f32, v2: f32) -> f32 {
        v1.powf(v2)
    }
    #[inline(always)]
    fn f64(v1: f64, v2: f64) -> f64 {
        v1.powf(v2)
    }
    #[inline(always)]
    fn u8(v1: u8, v2: u8) -> u8 {
        v1.pow(v2 as u32)
    }
    #[inline(always)]
    fn u16(v1: u16, v2: u16) -> u16 {
        v1.pow(v2 as u32)
    }
    #[inline(always)]
    fn u32(v1: u32, v2: u32) -> u32 {
        v1.pow(v2)
    }
    #[inline(always)]
    fn u64(v1: u64, v2: u64) -> u64 {
        v1.pow(v2 as u32)
    }
    #[inline(always)]
    fn i8(v1: i8, v2: i8) -> i8 {
        v1.pow(v2 as u32)
    }
    #[inline(always)]
    fn i16(v1: i16, v2: i16) -> i16 {
        v1.pow(v2 as u32)
    }
    #[inline(always)]
    fn i32(v1: i32, v2: i32) -> i32 {
        v1.pow(v2 as u32)
    }
    #[inline(always)]
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
    const KERNEL: &'static str = "binary_logical_and";

    #[inline(always)]
    fn bool(v1: bool, v2: bool) -> bool {
        v1 && v2
    }
    #[inline(always)]
    fn f8e4m3(v1: F8E4M3, v2: F8E4M3) -> bool {
        (v1 != F8E4M3::ZERO) && (v2 != F8E4M3::ZERO)
    }
    #[inline(always)]
    fn f8e5m2(v1: F8E5M2, v2: F8E5M2) -> bool {
        (v1 != F8E5M2::ZERO) && (v2 != F8E5M2::ZERO)
    }
    #[inline(always)]
    fn bf16(v1: bf16, v2: bf16) -> bool {
        (v1 != bf16::ZERO) && (v2 != bf16::ZERO)
    }
    #[inline(always)]
    fn f16(v1: f16, v2: f16) -> bool {
        (v1 != f16::ZERO) && (v2 != f16::ZERO)
    }
    #[inline(always)]
    fn f32(v1: f32, v2: f32) -> bool {
        (v1 != 0.0) && (v2 != 0.0)
    }
    #[inline(always)]
    fn f64(v1: f64, v2: f64) -> bool {
        (v1 != 0.0) && (v2 != 0.0)
    }
    #[inline(always)]
    fn u8(v1: u8, v2: u8) -> bool {
        (v1 != 0) && (v2 != 0)
    }
    #[inline(always)]
    fn u16(v1: u16, v2: u16) -> bool {
        (v1 != 0) && (v2 != 0)
    }
    #[inline(always)]
    fn u32(v1: u32, v2: u32) -> bool {
        (v1 != 0) && (v2 != 0)
    }
    #[inline(always)]
    fn u64(v1: u64, v2: u64) -> bool {
        (v1 != 0) && (v2 != 0)
    }
    #[inline(always)]
    fn i8(v1: i8, v2: i8) -> bool {
        (v1 != 0) && (v2 != 0)
    }
    #[inline(always)]
    fn i16(v1: i16, v2: i16) -> bool {
        (v1 != 0) && (v2 != 0)
    }
    #[inline(always)]
    fn i32(v1: i32, v2: i32) -> bool {
        (v1 != 0) && (v2 != 0)
    }
    #[inline(always)]
    fn i64(v1: i64, v2: i64) -> bool {
        (v1 != 0) && (v2 != 0)
    }
}
impl BinaryLogicalOpT for LogicalOr {
    const NAME: &'static str = "logical_or";
    const KERNEL: &'static str = "binary_logical_or";

    #[inline(always)]
    fn bool(v1: bool, v2: bool) -> bool {
        v1 || v2
    }
    #[inline(always)]
    fn f8e4m3(v1: F8E4M3, v2: F8E4M3) -> bool {
        (v1 != F8E4M3::ZERO) || (v2 != F8E4M3::ZERO)
    }
    #[inline(always)]
    fn f8e5m2(v1: F8E5M2, v2: F8E5M2) -> bool {
        (v1 != F8E5M2::ZERO) || (v2 != F8E5M2::ZERO)
    }
    #[inline(always)]
    fn bf16(v1: bf16, v2: bf16) -> bool {
        (v1 != bf16::ZERO) || (v2 != bf16::ZERO)
    }
    #[inline(always)]
    fn f16(v1: f16, v2: f16) -> bool {
        (v1 != f16::ZERO) || (v2 != f16::ZERO)
    }
    #[inline(always)]
    fn f32(v1: f32, v2: f32) -> bool {
        (v1 != 0.0) || (v2 != 0.0)
    }
    #[inline(always)]
    fn f64(v1: f64, v2: f64) -> bool {
        (v1 != 0.0) || (v2 != 0.0)
    }
    #[inline(always)]
    fn u8(v1: u8, v2: u8) -> bool {
        (v1 != 0) || (v2 != 0)
    }
    #[inline(always)]
    fn u16(v1: u16, v2: u16) -> bool {
        (v1 != 0) || (v2 != 0)
    }
    #[inline(always)]
    fn u32(v1: u32, v2: u32) -> bool {
        (v1 != 0) || (v2 != 0)
    }
    #[inline(always)]
    fn u64(v1: u64, v2: u64) -> bool {
        (v1 != 0) || (v2 != 0)
    }
    #[inline(always)]
    fn i8(v1: i8, v2: i8) -> bool {
        (v1 != 0) || (v2 != 0)
    }
    #[inline(always)]
    fn i16(v1: i16, v2: i16) -> bool {
        (v1 != 0) || (v2 != 0)
    }
    #[inline(always)]
    fn i32(v1: i32, v2: i32) -> bool {
        (v1 != 0) || (v2 != 0)
    }
    #[inline(always)]
    fn i64(v1: i64, v2: i64) -> bool {
        (v1 != 0) || (v2 != 0)
    }
}
impl BinaryLogicalOpT for LogicalXor {
    const NAME: &'static str = "logical_xor";
    const KERNEL: &'static str = "binary_logical_xor";

    #[inline(always)]
    fn bool(v1: bool, v2: bool) -> bool {
        v1 ^ v2
    }
    #[inline(always)]
    fn f8e4m3(v1: F8E4M3, v2: F8E4M3) -> bool {
        (v1 != F8E4M3::ZERO) ^ (v2 != F8E4M3::ZERO)
    }
    #[inline(always)]
    fn f8e5m2(v1: F8E5M2, v2: F8E5M2) -> bool {
        (v1 != F8E5M2::ZERO) ^ (v2 != F8E5M2::ZERO)
    }
    #[inline(always)]
    fn bf16(v1: bf16, v2: bf16) -> bool {
        (v1 != bf16::ZERO) ^ (v2 != bf16::ZERO)
    }
    #[inline(always)]
    fn f16(v1: f16, v2: f16) -> bool {
        (v1 != f16::ZERO) ^ (v2 != f16::ZERO)
    }
    #[inline(always)]
    fn f32(v1: f32, v2: f32) -> bool {
        (v1 != 0.0) ^ (v2 != 0.0)
    }
    #[inline(always)]
    fn f64(v1: f64, v2: f64) -> bool {
        (v1 != 0.0) ^ (v2 != 0.0)
    }
    #[inline(always)]
    fn u8(v1: u8, v2: u8) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
    #[inline(always)]
    fn u16(v1: u16, v2: u16) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
    #[inline(always)]
    fn u32(v1: u32, v2: u32) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
    #[inline(always)]
    fn u64(v1: u64, v2: u64) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
    #[inline(always)]
    fn i8(v1: i8, v2: i8) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
    #[inline(always)]
    fn i16(v1: i16, v2: i16) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
    #[inline(always)]
    fn i32(v1: i32, v2: i32) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
    #[inline(always)]
    fn i64(v1: i64, v2: i64) -> bool {
        (v1 != 0) ^ (v2 != 0)
    }
}

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
            const KERNEL: &'static str = concat!("cmp_", $name);

            #[inline(always)]
            fn bool(v1: bool, v2: bool) -> bool {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f8e4m3(v1: F8E4M3, v2: F8E4M3) -> bool {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f8e5m2(v1: F8E5M2, v2: F8E5M2) -> bool {
                $e(v1, v2)
            }
            #[inline(always)]
            fn bf16(v1: bf16, v2: bf16) -> bool {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f16(v1: f16, v2: f16) -> bool {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f32(v1: f32, v2: f32) -> bool {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f64(v1: f64, v2: f64) -> bool {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u8(v1: u8, v2: u8) -> bool {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u16(v1: u16, v2: u16) -> bool {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u32(v1: u32, v2: u32) -> bool {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u64(v1: u64, v2: u64) -> bool {
                $e(v1, v2)
            }
            #[inline(always)]
            fn i8(v1: i8, v2: i8) -> bool {
                $e(v1, v2)
            }
            #[inline(always)]
            fn i16(v1: i16, v2: i16) -> bool {
                $e(v1, v2)
            }
            #[inline(always)]
            fn i32(v1: i32, v2: i32) -> bool {
                $e(v1, v2)
            }
            #[inline(always)]
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
            const KERNEL: &'static str = concat!("cmp_scalar_", $name);

            #[inline(always)]
            fn bool(v1: bool, scalar: Scalar) -> bool {
                $e(v1, scalar.to_bool())
            }
            #[inline(always)]
            fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> bool {
                $e(v1, scalar.to_f8e4m3())
            }
            #[inline(always)]
            fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> bool {
                $e(v1, scalar.to_f8e5m2())
            }
            #[inline(always)]
            fn bf16(v1: bf16, scalar: Scalar) -> bool {
                $e(v1, scalar.to_bf16())
            }
            #[inline(always)]
            fn f16(v1: f16, scalar: Scalar) -> bool {
                $e(v1, scalar.to_f16())
            }
            #[inline(always)]
            fn f32(v1: f32, scalar: Scalar) -> bool {
                $e(v1, scalar.to_f32())
            }
            #[inline(always)]
            fn f64(v1: f64, scalar: Scalar) -> bool {
                $e(v1, scalar.to_f64())
            }
            #[inline(always)]
            fn u8(v1: u8, scalar: Scalar) -> bool {
                $e(v1, scalar.to_u8())
            }
            #[inline(always)]
            fn u16(v1: u16, scalar: Scalar) -> bool {
                $e(v1, scalar.to_u16())
            }
            #[inline(always)]
            fn u32(v1: u32, scalar: Scalar) -> bool {
                $e(v1, scalar.to_u32())
            }
            #[inline(always)]
            fn u64(v1: u64, scalar: Scalar) -> bool {
                $e(v1, scalar.to_u64())
            }
            #[inline(always)]
            fn i8(v1: i8, scalar: Scalar) -> bool {
                $e(v1, scalar.to_i8())
            }
            #[inline(always)]
            fn i16(v1: i16, scalar: Scalar) -> bool {
                $e(v1, scalar.to_i16())
            }
            #[inline(always)]
            fn i32(v1: i32, scalar: Scalar) -> bool {
                $e(v1, scalar.to_i32())
            }
            #[inline(always)]
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

pub(crate) struct Neg;
pub(crate) struct Abs;
pub(crate) struct Sign;
pub(crate) struct Square;
pub(crate) struct Relu;
pub(crate) struct Sigmoid;
pub(crate) struct Tanh;
pub(crate) struct Gelu;
pub(crate) struct Sin;
pub(crate) struct Cos;
pub(crate) struct Tan;
pub(crate) struct Ln;
pub(crate) struct Log10;
pub(crate) struct Log2;
pub(crate) struct Exp;
pub(crate) struct Exp10;
pub(crate) struct Exp2;
pub(crate) struct Softplus;
pub(crate) struct Recip;
pub(crate) struct Sqrt;

macro_rules! unary_op {
    ($op:ident, $name: literal, $a: ident, $e: expr) => {
        impl UnaryOpT for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("unary_", $name);

            #[inline(always)]
            fn bool(_: bool) -> bool {
                todo!("no unary function for bool")
            }
            #[inline(always)]
            fn f8e4m3($a: F8E4M3) -> F8E4M3 {
                $e
            }
            #[inline(always)]
            fn f8e5m2($a: F8E5M2) -> F8E5M2 {
                $e
            }
            #[inline(always)]
            fn bf16($a: bf16) -> bf16 {
                $e
            }
            #[inline(always)]
            fn f16($a: f16) -> f16 {
                $e
            }
            #[inline(always)]
            fn f32($a: f32) -> f32 {
                $e
            }
            #[inline(always)]
            fn f64($a: f64) -> f64 {
                $e
            }
            #[inline(always)]
            fn u8(_: u8) -> u8 {
                todo!("no unary function for u8")
            }
            #[inline(always)]
            fn u16(_: u16) -> u16 {
                todo!("no unary function for u16")
            }
            #[inline(always)]
            fn u32(_: u32) -> u32 {
                todo!("no unary function for u32")
            }
            #[inline(always)]
            fn u64(_: u64) -> u64 {
                todo!("no unary function for u64")
            }
            #[inline(always)]
            fn i8(_: i8) -> i8 {
                todo!("no unary function for i8")
            }
            #[inline(always)]
            fn i16(_: i16) -> i16 {
                todo!("no unary function for i16")
            }
            #[inline(always)]
            fn i32(_: i32) -> i32 {
                todo!("no unary function for i32")
            }
            #[inline(always)]
            fn i64(_: i64) -> i64 {
                todo!("no unary function for i64")
            }
        }
    };
}

unary_op!(Neg, "neg", v, -v);
unary_op!(Abs, "abs", v, v.abs());
impl UnaryOpT for Sign {
    const NAME: &'static str = "sign";
    const KERNEL: &'static str = "unary_sign";

    #[inline(always)]
    fn bool(_: bool) -> bool {
        todo!("no unary function for bool")
    }
    #[inline(always)]
    fn f8e4m3(v: F8E4M3) -> F8E4M3 {
        F8E4M3::from((v > F8E4M3::ZERO) as i8 as f32) - F8E4M3::from((v < F8E4M3::ZERO) as i8 as f32)
    }
    #[inline(always)]
    fn f8e5m2(v: F8E5M2) -> F8E5M2 {
        F8E5M2::from((v > F8E5M2::ZERO) as i8 as f32) - F8E5M2::from((v < F8E5M2::ZERO) as i8 as f32)
    }
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        bf16::from((v > bf16::ZERO) as i8) - bf16::from((v < bf16::ZERO) as i8)
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        f16::from((v > f16::ZERO) as i8) - f16::from((v < f16::ZERO) as i8)
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        f32::from(v > 0.) - f32::from(v < 0.)
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        f64::from(v > 0.) - f64::from(v < 0.)
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        todo!("no unary function for u8")
    }
    #[inline(always)]
    fn u16(_: u16) -> u16 {
        todo!("no unary function for u16")
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        todo!("no unary function for u32")
    }
    #[inline(always)]
    fn u64(_: u64) -> u64 {
        todo!("no unary function for u64")
    }
    #[inline(always)]
    fn i8(_: i8) -> i8 {
        todo!("no unary function for i8")
    }
    #[inline(always)]
    fn i16(_: i16) -> i16 {
        todo!("no unary function for i16")
    }
    #[inline(always)]
    fn i32(_: i32) -> i32 {
        todo!("no unary function for i32")
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        todo!("no unary function for i64")
    }
}
unary_op!(Square, "square", v, v * v);
impl UnaryOpT for Relu {
    const NAME: &'static str = "relu";
    const KERNEL: &'static str = "unary_relu";

    #[inline(always)]
    fn bool(_: bool) -> bool {
        todo!("no unary function for bool")
    }
    #[inline(always)]
    fn f8e4m3(v: F8E4M3) -> F8E4M3 {
        if v > F8E4M3::ZERO {
            v
        } else {
            F8E4M3::ZERO
        }
    }
    #[inline(always)]
    fn f8e5m2(v: F8E5M2) -> F8E5M2 {
        if v > F8E5M2::ZERO {
            v
        } else {
            F8E5M2::ZERO
        }
    }
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        if v > bf16::ZERO {
            v
        } else {
            bf16::ZERO
        }
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        if v > f16::ZERO {
            v
        } else {
            f16::ZERO
        }
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        if v > 0.0 {
            v
        } else {
            0.0
        }
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        if v > 0.0 {
            v
        } else {
            0.0
        }
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        todo!("no unary function for u8")
    }
    #[inline(always)]
    fn u16(_: u16) -> u16 {
        todo!("no unary function for u16")
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        todo!("no unary function for u32")
    }
    #[inline(always)]
    fn u64(_: u64) -> u64 {
        todo!("no unary function for u64")
    }
    #[inline(always)]
    fn i8(_: i8) -> i8 {
        todo!("no unary function for i8")
    }
    #[inline(always)]
    fn i16(_: i16) -> i16 {
        todo!("no unary function for i16")
    }
    #[inline(always)]
    fn i32(_: i32) -> i32 {
        todo!("no unary function for i32")
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        todo!("no unary function for i64")
    }
}
unary_op!(Sigmoid, "sigmoid", v, {
    let one = v / v;
    one / (one + (-v).exp())
});
unary_op!(Tanh, "tanh", v, v.tanh());
impl UnaryOpT for Gelu {
    const NAME: &'static str = "gelu";
    const KERNEL: &'static str = "unary_gelu";

    #[inline(always)]
    fn bool(_: bool) -> bool {
        todo!("no unary function for bool")
    }
    #[inline(always)]
    fn f8e4m3(v: F8E4M3) -> F8E4M3 {
        let half = F8E4M3::from(0.5f32);
        let one = F8E4M3::from(1.0f32);
        let sqrt_2_over_pi = F8E4M3::from(0.797_884_6_f32);
        let coeff = F8E4M3::from(0.044715f32);
        let x_cubed_term = v * v * v * coeff;
        v * half * (one + (sqrt_2_over_pi * (v + x_cubed_term)).tanh())
    }
    #[inline(always)]
    fn f8e5m2(v: F8E5M2) -> F8E5M2 {
        let half = F8E5M2::from(0.5f32);
        let one = F8E5M2::from(1.0f32);
        let sqrt_2_over_pi = F8E5M2::from(0.797_884_6_f32);
        let coeff = F8E5M2::from(0.044715f32);
        let x_cubed_term = v * v * v * coeff;
        v * half * (one + (sqrt_2_over_pi * (v + x_cubed_term)).tanh())
    }
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        let half = bf16::from_f32(0.5);
        let one = bf16::from_f32(1.0);
        let sqrt_2_over_pi = bf16::from_f32(0.797_884_6);
        let coeff = bf16::from_f32(0.044715);
        let x_cubed_term = v * v * v * coeff;
        v * half * (one + (sqrt_2_over_pi * (v + x_cubed_term)).tanh())
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        let half = f16::from_f32(0.5);
        let one = f16::from_f32(1.0);
        let sqrt_2_over_pi = f16::from_f32(0.797_884_6);
        let coeff = f16::from_f32(0.044715);
        let x_cubed_term = v * v * v * coeff;
        v * half * (one + (sqrt_2_over_pi * (v + x_cubed_term)).tanh())
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        let half = 0.5f32;
        let one = 1.0f32;
        let sqrt_2_over_pi = 0.797_884_6_f32;
        let coeff = 0.044715f32;
        let x_cubed_term = v * v * v * coeff;
        v * half * (one + (sqrt_2_over_pi * (v + x_cubed_term)).tanh())
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        let half = 0.5f64;
        let one = 1.0f64;
        let sqrt_2_over_pi = 0.7978845608028654f64;
        let coeff = 0.044715f64;
        let x_cubed_term = v * v * v * coeff;
        v * half * (one + (sqrt_2_over_pi * (v + x_cubed_term)).tanh())
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        todo!("no unary function for u8")
    }
    #[inline(always)]
    fn u16(_: u16) -> u16 {
        todo!("no unary function for u16")
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        todo!("no unary function for u32")
    }
    #[inline(always)]
    fn u64(_: u64) -> u64 {
        todo!("no unary function for u64")
    }
    #[inline(always)]
    fn i8(_: i8) -> i8 {
        todo!("no unary function for i8")
    }
    #[inline(always)]
    fn i16(_: i16) -> i16 {
        todo!("no unary function for i16")
    }
    #[inline(always)]
    fn i32(_: i32) -> i32 {
        todo!("no unary function for i32")
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        todo!("no unary function for i64")
    }
}
unary_op!(Sin, "sin", v, v.sin());
unary_op!(Cos, "cos", v, v.cos());
unary_op!(Tan, "tan", v, v.tan());
unary_op!(Ln, "ln", v, v.ln());
unary_op!(Log10, "log10", v, v.log10());
unary_op!(Log2, "log2", v, v.log2());
unary_op!(Exp, "exp", v, v.exp());
impl UnaryOpT for Exp10 {
    const NAME: &'static str = "exp10";
    const KERNEL: &'static str = "unary_exp10";

    #[inline(always)]
    fn bool(_: bool) -> bool {
        todo!("no unary function for bool")
    }
    #[inline(always)]
    fn f8e4m3(v: F8E4M3) -> F8E4M3 {
        let ten = F8E4M3::from(10.0f32);
        ten.powf(v)
    }
    #[inline(always)]
    fn f8e5m2(v: F8E5M2) -> F8E5M2 {
        let ten = F8E5M2::from(10.0f32);
        ten.powf(v)
    }
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        let ten = bf16::from_f32(10.0);
        ten.powf(v)
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        let ten = f16::from_f32(10.0);
        ten.powf(v)
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        let ten = 10.0f32;
        ten.powf(v)
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        let ten = 10.0f64;
        ten.powf(v)
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        todo!("no unary function for u8")
    }
    #[inline(always)]
    fn u16(_: u16) -> u16 {
        todo!("no unary function for u16")
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        todo!("no unary function for u32")
    }
    #[inline(always)]
    fn u64(_: u64) -> u64 {
        todo!("no unary function for u64")
    }
    #[inline(always)]
    fn i8(_: i8) -> i8 {
        todo!("no unary function for i8")
    }
    #[inline(always)]
    fn i16(_: i16) -> i16 {
        todo!("no unary function for i16")
    }
    #[inline(always)]
    fn i32(_: i32) -> i32 {
        todo!("no unary function for i32")
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        todo!("no unary function for i64")
    }
}
unary_op!(Exp2, "exp2", v, v.exp2());
unary_op!(Softplus, "softplus", v, (v.exp() + v / v).ln());
unary_op!(Recip, "recip", v, v.recip());
unary_op!(Sqrt, "sqrt", v, v.sqrt());

pub(crate) struct LogicalNot;

impl UnaryLogicalOpT for LogicalNot {
    const NAME: &'static str = "logical_not";
    const KERNEL: &'static str = "unary_logical_not";

    #[inline(always)]
    fn bool(v1: bool) -> bool {
        !v1
    }
    #[inline(always)]
    fn f8e4m3(v1: F8E4M3) -> bool {
        v1 == F8E4M3::ZERO
    }
    #[inline(always)]
    fn f8e5m2(v1: F8E5M2) -> bool {
        v1 == F8E5M2::ZERO
    }
    #[inline(always)]
    fn bf16(v1: bf16) -> bool {
        v1 == bf16::ZERO
    }
    #[inline(always)]
    fn f16(v1: f16) -> bool {
        v1 == f16::ZERO
    }
    #[inline(always)]
    fn f32(v1: f32) -> bool {
        v1 == 0.0
    }
    #[inline(always)]
    fn f64(v1: f64) -> bool {
        v1 == 0.0
    }
    #[inline(always)]
    fn u8(v1: u8) -> bool {
        v1 == 0
    }
    #[inline(always)]
    fn u16(v1: u16) -> bool {
        v1 == 0
    }
    #[inline(always)]
    fn u32(v1: u32) -> bool {
        v1 == 0
    }
    #[inline(always)]
    fn u64(v1: u64) -> bool {
        v1 == 0
    }
    #[inline(always)]
    fn i8(v1: i8) -> bool {
        v1 == 0
    }
    #[inline(always)]
    fn i16(v1: i16) -> bool {
        v1 == 0
    }
    #[inline(always)]
    fn i32(v1: i32) -> bool {
        v1 == 0
    }
    #[inline(always)]
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

macro_rules! unary_scalar_op {
    ($op:ident, $name: literal, $e: expr) => {
        impl UnaryScalarOpT for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("unary_scalar_", $name);

            #[inline(always)]
            fn bool(_: bool, _: Scalar) -> bool {
                todo!("no unary scalar function for bool")
            }
            #[inline(always)]
            fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> F8E4M3 {
                $e(v1, scalar.to_f8e4m3())
            }
            #[inline(always)]
            fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> F8E5M2 {
                $e(v1, scalar.to_f8e5m2())
            }
            #[inline(always)]
            fn bf16(v1: bf16, scalar: Scalar) -> bf16 {
                $e(v1, scalar.to_bf16())
            }
            #[inline(always)]
            fn f16(v1: f16, scalar: Scalar) -> f16 {
                $e(v1, scalar.to_f16())
            }
            #[inline(always)]
            fn f32(v1: f32, scalar: Scalar) -> f32 {
                $e(v1, scalar.to_f32())
            }
            #[inline(always)]
            fn f64(v1: f64, scalar: Scalar) -> f64 {
                $e(v1, scalar.to_f64())
            }
            #[inline(always)]
            fn u8(_: u8, _: Scalar) -> u8 {
                todo!("no unary scalar function for u8")
            }
            #[inline(always)]
            fn u16(_: u16, _: Scalar) -> u16 {
                todo!("no unary scalar function for u16")
            }
            #[inline(always)]
            fn u32(_: u32, _: Scalar) -> u32 {
                todo!("no unary scalar function for u32")
            }
            #[inline(always)]
            fn u64(_: u64, _: Scalar) -> u64 {
                todo!("no unary scalar function for u64")
            }
            #[inline(always)]
            fn i8(_: i8, _: Scalar) -> i8 {
                todo!("no unary scalar function for i8")
            }
            #[inline(always)]
            fn i16(_: i16, _: Scalar) -> i16 {
                todo!("no unary scalar function for i16")
            }
            #[inline(always)]
            fn i32(_: i32, _: Scalar) -> i32 {
                todo!("no unary scalar function for i32")
            }
            #[inline(always)]
            fn i64(_: i64, _: Scalar) -> i64 {
                todo!("no unary scalar function for i64")
            }
        }
    };
}

unary_scalar_op!(AddScalar, "add_scalar", |v1, v2| v1 + v2);
unary_scalar_op!(SubScalar, "sub_scalar", |v1, v2| v1 - v2);
unary_scalar_op!(MulScalar, "mul_scalar", |v1, v2| v1 * v2);
unary_scalar_op!(DivScalar, "div_scalar", |v1, v2| v1 / v2);
impl UnaryScalarOpT for PowScalar {
    const NAME: &'static str = "pow_scalar";
    const KERNEL: &'static str = "unary_scalar_pow_scalar";

    #[inline(always)]
    fn bool(_: bool, _: Scalar) -> bool {
        todo!("no unary scalar function for bool")
    }
    #[inline(always)]
    fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> F8E4M3 {
        v1.powf(scalar.to_f8e4m3())
    }
    #[inline(always)]
    fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> F8E5M2 {
        v1.powf(scalar.to_f8e5m2())
    }
    #[inline(always)]
    fn bf16(v1: bf16, scalar: Scalar) -> bf16 {
        v1.powf(scalar.to_bf16())
    }
    #[inline(always)]
    fn f16(v1: f16, scalar: Scalar) -> f16 {
        v1.powf(scalar.to_f16())
    }
    #[inline(always)]
    fn f32(v1: f32, scalar: Scalar) -> f32 {
        v1.powf(scalar.to_f32())
    }
    #[inline(always)]
    fn f64(v1: f64, scalar: Scalar) -> f64 {
        v1.powf(scalar.to_f64())
    }
    #[inline(always)]
    fn u8(v1: u8, scalar: Scalar) -> u8 {
        v1.pow(scalar.to_u32())
    }
    #[inline(always)]
    fn u16(v1: u16, scalar: Scalar) -> u16 {
        v1.pow(scalar.to_u32())
    }
    #[inline(always)]
    fn u32(v1: u32, scalar: Scalar) -> u32 {
        v1.pow(scalar.to_u32())
    }
    #[inline(always)]
    fn u64(v1: u64, scalar: Scalar) -> u64 {
        v1.pow(scalar.to_u32())
    }
    #[inline(always)]
    fn i8(v1: i8, scalar: Scalar) -> i8 {
        v1.pow(scalar.to_u32())
    }
    #[inline(always)]
    fn i16(v1: i16, scalar: Scalar) -> i16 {
        v1.pow(scalar.to_u32())
    }
    #[inline(always)]
    fn i32(v1: i32, scalar: Scalar) -> i32 {
        v1.pow(scalar.to_u32())
    }
    #[inline(always)]
    fn i64(v1: i64, scalar: Scalar) -> i64 {
        v1.pow(scalar.to_u32())
    }
}
unary_scalar_op!(MaximumScalar, "maximum_scalar", |v1, v2| if v1 < v2 { v2 } else { v1 });
unary_scalar_op!(MinimumScalar, "minimum_scalar", |v1, v2| if v1 > v2 { v2 } else { v1 });
impl UnaryScalarOpT for LeakyRelu {
    const NAME: &'static str = "leaky_relu";
    const KERNEL: &'static str = "unary_scalar_leaky_relu";

    #[inline(always)]
    fn bool(_: bool, _: Scalar) -> bool {
        todo!("no unary scalar function for bool")
    }
    #[inline(always)]
    fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> F8E4M3 {
        let alpha = scalar.to_f8e4m3();
        if v1 > F8E4M3::ZERO {
            v1
        } else {
            v1 * alpha
        }
    }
    #[inline(always)]
    fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> F8E5M2 {
        let alpha = scalar.to_f8e5m2();
        if v1 > F8E5M2::ZERO {
            v1
        } else {
            v1 * alpha
        }
    }
    #[inline(always)]
    fn bf16(v1: bf16, scalar: Scalar) -> bf16 {
        let alpha = scalar.to_bf16();
        if v1 > bf16::ZERO {
            v1
        } else {
            v1 * alpha
        }
    }
    #[inline(always)]
    fn f16(v1: f16, scalar: Scalar) -> f16 {
        let alpha = scalar.to_f16();
        if v1 > f16::ZERO {
            v1
        } else {
            v1 * alpha
        }
    }
    #[inline(always)]
    fn f32(v1: f32, scalar: Scalar) -> f32 {
        let alpha = scalar.to_f32();
        if v1 > 0.0 {
            v1
        } else {
            v1 * alpha
        }
    }
    #[inline(always)]
    fn f64(v1: f64, scalar: Scalar) -> f64 {
        let alpha = scalar.to_f64();
        if v1 > 0.0 {
            v1
        } else {
            v1 * alpha
        }
    }
    #[inline(always)]
    fn u8(_: u8, _: Scalar) -> u8 {
        todo!("no unary scalar function for u8")
    }
    #[inline(always)]
    fn u16(_: u16, _: Scalar) -> u16 {
        todo!("no unary scalar function for u16")
    }
    #[inline(always)]
    fn u32(_: u32, _: Scalar) -> u32 {
        todo!("no unary scalar function for u32")
    }
    #[inline(always)]
    fn u64(_: u64, _: Scalar) -> u64 {
        todo!("no unary scalar function for u64")
    }
    #[inline(always)]
    fn i8(_: i8, _: Scalar) -> i8 {
        todo!("no unary scalar function for i8")
    }
    #[inline(always)]
    fn i16(_: i16, _: Scalar) -> i16 {
        todo!("no unary scalar function for i16")
    }
    #[inline(always)]
    fn i32(_: i32, _: Scalar) -> i32 {
        todo!("no unary scalar function for i32")
    }
    #[inline(always)]
    fn i64(_: i64, _: Scalar) -> i64 {
        todo!("no unary scalar function for i64")
    }
}
impl UnaryScalarOpT for Elu {
    const NAME: &'static str = "elu";
    const KERNEL: &'static str = "unary_scalar_elu";

    #[inline(always)]
    fn bool(_: bool, _: Scalar) -> bool {
        todo!("no unary scalar function for bool")
    }
    #[inline(always)]
    fn f8e4m3(v1: F8E4M3, scalar: Scalar) -> F8E4M3 {
        let alpha = scalar.to_f8e4m3();
        if v1 > F8E4M3::ZERO {
            v1
        } else {
            alpha * (v1.exp() - F8E4M3::ONE)
        }
    }
    #[inline(always)]
    fn f8e5m2(v1: F8E5M2, scalar: Scalar) -> F8E5M2 {
        let alpha = scalar.to_f8e5m2();
        if v1 > F8E5M2::ZERO {
            v1
        } else {
            alpha * (v1.exp() - F8E5M2::ONE)
        }
    }
    #[inline(always)]
    fn bf16(v1: bf16, scalar: Scalar) -> bf16 {
        let alpha = scalar.to_bf16();
        if v1 > bf16::ZERO {
            v1
        } else {
            alpha * (v1.exp() - bf16::ONE)
        }
    }
    #[inline(always)]
    fn f16(v1: f16, scalar: Scalar) -> f16 {
        let alpha = scalar.to_f16();
        if v1 > f16::ZERO {
            v1
        } else {
            alpha * (v1.exp() - f16::ONE)
        }
    }
    #[inline(always)]
    fn f32(v1: f32, scalar: Scalar) -> f32 {
        let alpha = scalar.to_f32();
        if v1 > 0.0 {
            v1
        } else {
            alpha * (v1.exp() - 1.0)
        }
    }
    #[inline(always)]
    fn f64(v1: f64, scalar: Scalar) -> f64 {
        let alpha = scalar.to_f64();
        if v1 > 0.0 {
            v1
        } else {
            alpha * (v1.exp() - 1.0)
        }
    }
    #[inline(always)]
    fn u8(_: u8, _: Scalar) -> u8 {
        todo!("no unary scalar function for u8")
    }
    #[inline(always)]
    fn u16(_: u16, _: Scalar) -> u16 {
        todo!("no unary scalar function for u16")
    }
    #[inline(always)]
    fn u32(_: u32, _: Scalar) -> u32 {
        todo!("no unary scalar function for u32")
    }
    #[inline(always)]
    fn u64(_: u64, _: Scalar) -> u64 {
        todo!("no unary scalar function for u64")
    }
    #[inline(always)]
    fn i8(_: i8, _: Scalar) -> i8 {
        todo!("no unary scalar function for i8")
    }
    #[inline(always)]
    fn i16(_: i16, _: Scalar) -> i16 {
        todo!("no unary scalar function for i16")
    }
    #[inline(always)]
    fn i32(_: i32, _: Scalar) -> i32 {
        todo!("no unary scalar function for i32")
    }
    #[inline(always)]
    fn i64(_: i64, _: Scalar) -> i64 {
        todo!("no unary scalar function for i64")
    }
}
