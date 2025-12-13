use crate::{
    scalar::Scalar,
    tensor::TensorId,
    types::{DType, DynamicDimId},
};

// Binary Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BinaryParams;

// BinaryLogical Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BinaryLogicalParams;

// Cmp Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CmpParams;

// CmpScalar Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CmpScalarParams {
    pub scalar: Scalar,
}

// Unary Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnaryParams;

// UnaryLogical Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnaryLogicalParams;

// UnaryScalar Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnaryScalarParams {
    pub scalar: Scalar,
}

// Matrix Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MatmulParams;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DotParams;

// Reduce Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReduceParams {
    pub dims: Vec<Scalar>,
    pub keep_dim: bool,
}

// Concat Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConcatParams {
    pub dim: Scalar,
}

// Split Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SplitParams {
    pub dim: Scalar,
    pub sizes: Vec<Scalar>,
    pub output_index: usize,
}

// Indexing Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IndexSelectParams {
    pub dim: Scalar,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IndexPutParams {
    pub dim: Scalar,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GatherParams {
    pub dim: Scalar,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScatterParams {
    pub dim: Scalar,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScatterAddParams {
    pub dim: Scalar,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScatterMaxParams {
    pub dim: Scalar,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScatterMinParams {
    pub dim: Scalar,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OnehotoParams {
    pub num_classes: Scalar,
    pub axis: Scalar,
    pub dtype: DType,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NonzeroParams {
    /// Dynamic dimension ID for the count of nonzero elements (N in [N, ndim])
    pub dynamic_count_dim: Option<DynamicDimId>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UniqueParams {
    pub inverse_id: TensorId,
    pub counts_id: TensorId,
    /// Dynamic dimension ID for the count of unique elements (M in values[M], counts[M])
    pub dynamic_count_dim: Option<DynamicDimId>,
}

// Conv Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Conv1dParams {
    pub batch_size: usize,
    pub length_input: usize,
    pub channels_output: usize,
    pub channels_input: usize,
    pub kernel_size: usize,
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Conv2dParams {
    pub batch_size: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
    pub channels_output: usize,
    pub channels_input: usize,
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Conv3dParams {
    pub batch_size: usize,
    pub input_depth: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub kernel_depth: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
    pub channels_output: usize,
    pub channels_input: usize,
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvTranspose1dParams {
    pub batch_size: usize,
    pub length_input: usize,
    pub channels_output: usize,
    pub channels_input: usize,
    pub kernel_size: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub stride: usize,
    pub dilation: usize,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvTranspose2dParams {
    pub batch_size: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
    pub channels_output: usize,
    pub channels_input: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub stride: usize,
    pub dilation: usize,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvTranspose3dParams {
    pub batch_size: usize,
    pub input_depth: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub kernel_depth: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
    pub channels_output: usize,
    pub channels_input: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub stride: usize,
    pub dilation: usize,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Conv1dGradWeightParams {
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out_channels: usize,
    pub in_channels: usize,
    pub kernel_size: usize,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Conv2dGradWeightParams {
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out_channels: usize,
    pub in_channels: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Conv3dGradWeightParams {
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out_channels: usize,
    pub in_channels: usize,
    pub kernel_depth: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvTranspose1dGradWeightParams {
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvTranspose2dGradWeightParams {
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvTranspose3dGradWeightParams {
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_depth: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
}

// Windowing Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReduceWindowParams {
    pub window_shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub padding: Vec<(usize, usize)>, // per-dimension: [(lo, hi), ...]
    pub aux_tensors: Vec<TensorId>,   // for storing indices from max/min pooling
}

// Padding Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PaddingParams {
    pub padding: Vec<(usize, usize)>, // per-dimension: [(before, after), ...]
    pub pad_value: Scalar,
}

// Scan Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScanParams {
    pub dim: usize,
}

// Sort Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TopKParams {
    pub k: usize,
    pub dim: i32,
    pub largest: bool,
    pub sorted: bool,
    pub indices_id: TensorId,
}

// Einsum Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EinsumParams {
    pub equation: String,
    pub input_subscripts: Vec<Vec<char>>,
    pub output_subscripts: Vec<char>,
    pub contraction_indices: Vec<char>,
}

// Resize Operations

/// Interpolation mode for resize operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResizeMode {
    #[default]
    Nearest,
    Linear, // bilinear (2D) / trilinear (3D)
    Cubic,  // bicubic (2D only)
}

/// Coordinate transformation mode (ONNX spec)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResizeCoordTransform {
    #[default]
    HalfPixel,
    Asymmetric,
    AlignCorners,
    PytorchHalfPixel,
}

/// Rounding mode for nearest neighbor interpolation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResizeNearestMode {
    #[default]
    Floor,
    Ceil,
    RoundPreferFloor,
    RoundPreferCeil,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResizeParams {
    pub output_size: Vec<usize>,
    pub mode: ResizeMode,
    pub coord_transform: ResizeCoordTransform,
    pub nearest_mode: ResizeNearestMode,
}

// Shape Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReshapeParams;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FlattenParams;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SqueezeParams;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnsqueezeParams;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BroadcastParams;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TransposeParams;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PermuteParams;

// Shape Memory Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FlipParams {
    pub dims: Vec<usize>,
}

// Shape Scalar Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SliceParams {
    pub dim: Scalar,
    pub start: Scalar,
    pub end: Scalar,
    pub step: Scalar,
}

// Cast Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ToDTypeParams {
    pub dtype: DType,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ToDeviceParams;

// Memory Operations

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContiguousParams;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SetParams;

// OpParams Enum

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OpParams {
    // Binary
    Binary(BinaryParams),
    BinaryLogical(BinaryLogicalParams),

    // Cmp
    Cmp(CmpParams),
    CmpScalar(CmpScalarParams),

    // Unary
    Unary(UnaryParams),
    UnaryLogical(UnaryLogicalParams),
    UnaryScalar(UnaryScalarParams),

    // Matrix
    Matmul(MatmulParams),
    Dot(DotParams),

    // Reduce
    Reduce(ReduceParams),

    // Concat
    Concat(ConcatParams),

    // Split
    Split(SplitParams),

    // Indexing
    IndexSelect(IndexSelectParams),
    IndexPut(IndexPutParams),
    Gather(GatherParams),
    Scatter(ScatterParams),
    ScatterAdd(ScatterAddParams),
    ScatterMax(ScatterMaxParams),
    ScatterMin(ScatterMinParams),
    Onehoto(OnehotoParams),
    Nonzero(NonzeroParams),
    Unique(UniqueParams),

    // Conv
    Conv1d(Conv1dParams),
    Conv2d(Conv2dParams),
    Conv3d(Conv3dParams),
    ConvTranspose1d(ConvTranspose1dParams),
    ConvTranspose2d(ConvTranspose2dParams),
    ConvTranspose3d(ConvTranspose3dParams),
    Conv1dGradWeight(Conv1dGradWeightParams),
    Conv2dGradWeight(Conv2dGradWeightParams),
    Conv3dGradWeight(Conv3dGradWeightParams),
    ConvTranspose1dGradWeight(ConvTranspose1dGradWeightParams),
    ConvTranspose2dGradWeight(ConvTranspose2dGradWeightParams),
    ConvTranspose3dGradWeight(ConvTranspose3dGradWeightParams),

    // Windowing
    ReduceWindow(ReduceWindowParams),

    // Padding
    Padding(PaddingParams),

    // Scan
    Scan(ScanParams),

    // Sort
    TopK(TopKParams),

    // Einsum
    Einsum(EinsumParams),

    // Resize
    Resize(ResizeParams),

    // Shape
    Reshape(ReshapeParams),
    Flatten(FlattenParams),
    Squeeze(SqueezeParams),
    Unsqueeze(UnsqueezeParams),
    Broadcast(BroadcastParams),
    Transpose(TransposeParams),
    Permute(PermuteParams),

    // Shape Memory
    Flip(FlipParams),

    // Shape Scalar
    Slice(SliceParams),

    // Cast
    ToDType(ToDTypeParams),
    ToDevice(ToDeviceParams),

    // Memory
    Contiguous(ContiguousParams),
    Set(SetParams),
}
