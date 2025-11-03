#![allow(clippy::bool_comparison, clippy::eq_op)]

// Naming convention for kernel names to avoid duplication:
// BinaryLogicalOp -> binary_logical_and (not binary_logical_logical_and)
// UnaryLogicalOp -> unary_logical_not (not unary_logical_logical_not)

mod binary;
mod cmp;
pub mod conv;
mod unary;
pub mod utils;
pub mod window_reduction;

use crate::{compat::*, scalar::Scalar, tensor::TensorId};
pub(crate) use binary::*;
pub(crate) use cmp::*;
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};
pub(crate) use unary::*;

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
    Silu,
    Mish,

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
    Prelu,
}

pub trait UnaryScalarOpT {
    const NAME: &'static str;

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
    Any,    // no-backprop
    All,    // no-backprop
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Mean => "mean",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
            ReduceOp::Prod => "prod",
            ReduceOp::Std => "std",
            ReduceOp::Var => "var",
            ReduceOp::Norm => "norm",
            ReduceOp::ArgMax => "argmax",
            ReduceOp::ArgMin => "argmin",
            ReduceOp::Any => "any",
            ReduceOp::All => "all",
        };
        write!(f, "{}", s)
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
    IndexPut,
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
pub enum WindowingOp {
    ReduceWindow,
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
pub enum ShapeScalarsOp {
    Slice,
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
    Reduce(ReduceOp, TensorId, bool, Vec<Scalar>),
    Concat(ConcatOp, Vec<TensorId>, Vec<Scalar>),
    Split(SplitOp, TensorId, Vec<Scalar>, usize),
    Indexing(IndexingOp, Vec<TensorId>, Vec<Scalar>),
    Conv(ConvOp, TensorId, TensorId, Vec<Scalar>),
    Windowing(WindowingOp, TensorId, Vec<Scalar>),
    Shape(ShapeOp, TensorId),
    ShapeScalars(ShapeScalarsOp, TensorId, Vec<Scalar>),
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
            Op::Reduce(_, t, _, _) => vec![*t],
            Op::Concat(_, tt, _) => tt.clone(),
            Op::Split(_, t, _, _) => vec![*t],
            Op::Indexing(_, tt, _) => tt.clone(),
            Op::Conv(_, input, weight, _) => vec![*input, *weight],
            Op::Windowing(_, t, _) => vec![*t],
            Op::Shape(_, t) => vec![*t],
            Op::ShapeScalars(_, t, _) => vec![*t],
            Op::Cast(_, t) => vec![*t],
            Op::Memory(_, t) => vec![*t],
            Op::Dummy => vec![],
        }
    }
}
