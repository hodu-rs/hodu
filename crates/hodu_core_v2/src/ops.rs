// Naming convention for kernel names to avoid duplication:
// BinaryLogicalOp -> binary_logical_and (not binary_logical_logical_and)
// UnaryLogicalOp -> unary_logical_not (not unary_logical_logical_not)

use crate::layer::compat::fmt;

#[derive(Clone, Copy, PartialEq, Eq)]
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

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "add"),
            Self::Sub => write!(f, "sub"),
            Self::Mul => write!(f, "mul"),
            Self::Div => write!(f, "div"),
            Self::Pow => write!(f, "pow"),
            Self::Maximum => write!(f, "maximum"),
            Self::Minimum => write!(f, "minimum"),
        }
    }
}

impl fmt::Debug for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum BinaryLogicalOp {
    LogicalAnd, // no-backprop
    LogicalOr,  // no-backprop
    LogicalXor, // no-backprop
}

impl fmt::Display for BinaryLogicalOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LogicalAnd => write!(f, "logical_and"),
            Self::LogicalOr => write!(f, "logical_or"),
            Self::LogicalXor => write!(f, "logical_xor"),
        }
    }
}

impl fmt::Debug for BinaryLogicalOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
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

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eq => write!(f, "eq"),
            Self::Ne => write!(f, "ne"),
            Self::Lt => write!(f, "lt"),
            Self::Le => write!(f, "le"),
            Self::Gt => write!(f, "gt"),
            Self::Ge => write!(f, "ge"),
        }
    }
}

impl fmt::Debug for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
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

impl fmt::Display for CmpScalarOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EqScalar => write!(f, "eq_scalar"),
            Self::NeScalar => write!(f, "ne_scalar"),
            Self::LtScalar => write!(f, "lt_scalar"),
            Self::LeScalar => write!(f, "le_scalar"),
            Self::GtScalar => write!(f, "gt_scalar"),
            Self::GeScalar => write!(f, "ge_scalar"),
        }
    }
}

impl fmt::Debug for CmpScalarOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
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

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Neg => write!(f, "neg"),
            Self::Abs => write!(f, "abs"),
            Self::Sign => write!(f, "sign"),
            Self::Square => write!(f, "square"),
            Self::Sqrt => write!(f, "sqrt"),
            Self::Recip => write!(f, "recip"),
            Self::Relu => write!(f, "relu"),
            Self::Sigmoid => write!(f, "sigmoid"),
            Self::Tanh => write!(f, "tanh"),
            Self::Gelu => write!(f, "gelu"),
            Self::Softplus => write!(f, "softplus"),
            Self::Silu => write!(f, "silu"),
            Self::Mish => write!(f, "mish"),
            Self::Sin => write!(f, "sin"),
            Self::Cos => write!(f, "cos"),
            Self::Tan => write!(f, "tan"),
            Self::Exp => write!(f, "exp"),
            Self::Exp2 => write!(f, "exp2"),
            Self::Exp10 => write!(f, "exp10"),
            Self::Ln => write!(f, "ln"),
            Self::Log2 => write!(f, "log2"),
            Self::Log10 => write!(f, "log10"),
        }
    }
}

impl fmt::Debug for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum UnaryLogicalOp {
    LogicalNot, // no-backprop
}

impl fmt::Display for UnaryLogicalOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LogicalNot => write!(f, "logical_not"),
        }
    }
}

impl fmt::Debug for UnaryLogicalOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
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

impl fmt::Display for UnaryScalarOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AddScalar => write!(f, "add_scalar"),
            Self::SubScalar => write!(f, "sub_scalar"),
            Self::MulScalar => write!(f, "mul_scalar"),
            Self::DivScalar => write!(f, "div_scalar"),
            Self::PowScalar => write!(f, "pow_scalar"),
            Self::MaximumScalar => write!(f, "maximum_scalar"),
            Self::MinimumScalar => write!(f, "minimum_scalar"),
            Self::LeakyRelu => write!(f, "leaky_relu"),
            Self::Elu => write!(f, "elu"),
            Self::Prelu => write!(f, "prelu"),
        }
    }
}

impl fmt::Debug for UnaryScalarOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum MatrixOp {
    Matmul,
    Dot, // Supports 1D dot product, 2D matmul, and ND broadcast matmul
}

impl fmt::Display for MatrixOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matmul => write!(f, "matmul"),
            Self::Dot => write!(f, "dot"),
        }
    }
}

impl fmt::Debug for MatrixOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
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
        match self {
            Self::Sum => write!(f, "sum"),
            Self::Mean => write!(f, "mean"),
            Self::Max => write!(f, "max"),
            Self::Min => write!(f, "min"),
            Self::Prod => write!(f, "prod"),
            Self::Std => write!(f, "std"),
            Self::Var => write!(f, "var"),
            Self::Norm => write!(f, "norm"),
            Self::ArgMax => write!(f, "argmax"),
            Self::ArgMin => write!(f, "argmin"),
            Self::Any => write!(f, "any"),
            Self::All => write!(f, "all"),
        }
    }
}

impl fmt::Debug for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum ConcatOp {
    Concat,
    Stack,
}

impl fmt::Display for ConcatOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Concat => write!(f, "concat"),
            Self::Stack => write!(f, "stack"),
        }
    }
}

impl fmt::Debug for ConcatOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum SplitOp {
    Split,
    Chunk,
}

impl fmt::Display for SplitOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Split => write!(f, "split"),
            Self::Chunk => write!(f, "chunk"),
        }
    }
}

impl fmt::Debug for SplitOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
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

impl fmt::Display for IndexingOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexSelect => write!(f, "index_select"),
            Self::IndexPut => write!(f, "index_put"),
            Self::Gather => write!(f, "gather"),
            Self::Scatter => write!(f, "scatter"),
            Self::ScatterAdd => write!(f, "scatter_add"),
            Self::ScatterMax => write!(f, "scatter_max"),
            Self::ScatterMin => write!(f, "scatter_min"),
        }
    }
}

impl fmt::Debug for IndexingOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
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

impl fmt::Display for ConvOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conv1d => write!(f, "conv1d"),
            Self::Conv2d => write!(f, "conv2d"),
            Self::Conv3d => write!(f, "conv3d"),
            Self::ConvTranspose1d => write!(f, "conv_transpose1d"),
            Self::ConvTranspose2d => write!(f, "conv_transpose2d"),
            Self::ConvTranspose3d => write!(f, "conv_transpose3d"),
            Self::Conv1dGradWeight => write!(f, "conv1d_grad_weight"),
            Self::Conv2dGradWeight => write!(f, "conv2d_grad_weight"),
            Self::Conv3dGradWeight => write!(f, "conv3d_grad_weight"),
            Self::ConvTranspose1dGradWeight => write!(f, "conv_transpose1d_grad_weight"),
            Self::ConvTranspose2dGradWeight => write!(f, "conv_transpose2d_grad_weight"),
            Self::ConvTranspose3dGradWeight => write!(f, "conv_transpose3d_grad_weight"),
        }
    }
}

impl fmt::Debug for ConvOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum WindowingOp {
    ReduceWindowMax,
    ReduceWindowMean,
    ReduceWindowSum,
    ReduceWindowMin,
}

impl fmt::Display for WindowingOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReduceWindowMax => write!(f, "reduce_window_max"),
            Self::ReduceWindowMean => write!(f, "reduce_window_mean"),
            Self::ReduceWindowSum => write!(f, "reduce_window_sum"),
            Self::ReduceWindowMin => write!(f, "reduce_window_min"),
        }
    }
}

impl fmt::Debug for WindowingOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
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

impl fmt::Display for ShapeOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reshape => write!(f, "reshape"),
            Self::Flatten => write!(f, "flatten"),
            Self::Squeeze => write!(f, "squeeze"),
            Self::Unsqueeze => write!(f, "unsqueeze"),
            Self::Broadcast => write!(f, "broadcast"),
            Self::Transpose => write!(f, "transpose"),
            Self::Permute => write!(f, "permute"),
        }
    }
}

impl fmt::Debug for ShapeOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum ShapeScalarsOp {
    Slice,
}

impl fmt::Display for ShapeScalarsOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Slice => write!(f, "slice"),
        }
    }
}

impl fmt::Debug for ShapeScalarsOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum CastOp {
    ToDType, // no-backprop
}

impl fmt::Display for CastOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ToDType => write!(f, "to_dtype"),
        }
    }
}

impl fmt::Debug for CastOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum MemoryOp {
    Contiguous, // no-backprop
}

impl fmt::Display for MemoryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contiguous => write!(f, "contiguous"),
        }
    }
}

impl fmt::Debug for MemoryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum Op {
    Binary(BinaryOp),
    BinaryLogical(BinaryLogicalOp),
    Cmp(CmpOp),
    CmpScalar(CmpScalarOp),
    Unary(UnaryOp),
    UnaryLogical(UnaryLogicalOp),
    UnaryScalar(UnaryScalarOp),
    Matrix(MatrixOp),
    Reduce(ReduceOp),
    Concat(ConcatOp),
    Split(SplitOp),
    Indexing(IndexingOp),
    Conv(ConvOp),
    Windowing(WindowingOp),
    Shape(ShapeOp),
    ShapeScalars(ShapeScalarsOp),
    Cast(CastOp),
    Memory(MemoryOp),
    Dummy,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binary(op) => write!(f, "{}", op),
            Self::BinaryLogical(op) => write!(f, "{}", op),
            Self::Cmp(op) => write!(f, "{}", op),
            Self::CmpScalar(op) => write!(f, "{}", op),
            Self::Unary(op) => write!(f, "{}", op),
            Self::UnaryLogical(op) => write!(f, "{}", op),
            Self::UnaryScalar(op) => write!(f, "{}", op),
            Self::Matrix(op) => write!(f, "{}", op),
            Self::Reduce(op) => write!(f, "{}", op),
            Self::Concat(op) => write!(f, "{}", op),
            Self::Split(op) => write!(f, "{}", op),
            Self::Indexing(op) => write!(f, "{}", op),
            Self::Conv(op) => write!(f, "{}", op),
            Self::Windowing(op) => write!(f, "{}", op),
            Self::Shape(op) => write!(f, "{}", op),
            Self::ShapeScalars(op) => write!(f, "{}", op),
            Self::Cast(op) => write!(f, "{}", op),
            Self::Memory(op) => write!(f, "{}", op),
            Self::Dummy => write!(f, "dummy"),
        }
    }
}

impl fmt::Debug for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binary(op) => write!(f, "Binary[{}]", op),
            Self::BinaryLogical(op) => write!(f, "BinaryLogical[{}]", op),
            Self::Cmp(op) => write!(f, "Cmp[{}]", op),
            Self::CmpScalar(op) => write!(f, "CmpScalar[{}]", op),
            Self::Unary(op) => write!(f, "Unary[{}]", op),
            Self::UnaryLogical(op) => write!(f, "UnaryLogical[{}]", op),
            Self::UnaryScalar(op) => write!(f, "UnaryScalar[{}]", op),
            Self::Matrix(op) => write!(f, "Matrix[{}]", op),
            Self::Reduce(op) => write!(f, "Reduce[{}]", op),
            Self::Concat(op) => write!(f, "Concat[{}]", op),
            Self::Split(op) => write!(f, "Split[{}]", op),
            Self::Indexing(op) => write!(f, "Indexing[{}]", op),
            Self::Conv(op) => write!(f, "Conv[{}]", op),
            Self::Windowing(op) => write!(f, "Windowing[{}]", op),
            Self::Shape(op) => write!(f, "Shape[{}]", op),
            Self::ShapeScalars(op) => write!(f, "ShapeScalars[{}]", op),
            Self::Cast(op) => write!(f, "Cast[{}]", op),
            Self::Memory(op) => write!(f, "Memory[{}]", op),
            Self::Dummy => write!(f, "Dummy"),
        }
    }
}
