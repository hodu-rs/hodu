//! Interpreter Runtime - executes Snapshot IR directly without compilation
//!
//! This is a builtin runtime that interprets the Snapshot graph node-by-node,
//! delegating actual tensor operations to hodu_core's CPU backend.

use crate::{
    CompiledArtifact, Device, ExecutableModule, ExecutableModuleInner, HoduError, HoduResult, OutputFormat,
    RuntimePlugin, Tensor,
};
use float8::F8E4M3;
#[cfg(feature = "f8e5m2")]
use float8::F8E5M2;
use half::{bf16, f16};
use hodu_compat::*;
use hodu_core::ops::*;
use hodu_core::script::{Snapshot, SnapshotConstant, SnapshotNode, SnapshotTensorId};
use hodu_core::types::DType;
use std::path::Path;

/// Interpreter runtime plugin
pub struct InterpRuntime;

impl InterpRuntime {
    pub fn new() -> Self {
        Self
    }
}

impl Default for InterpRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimePlugin for InterpRuntime {
    fn name(&self) -> &str {
        "interp"
    }

    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::CPU]
    }

    fn loadable_formats(&self, _device: Device) -> Vec<OutputFormat> {
        vec![OutputFormat::HoduSnapshot]
    }

    fn load(&self, artifact: &CompiledArtifact, _device: Device) -> HoduResult<ExecutableModule> {
        match artifact.format {
            OutputFormat::HoduSnapshot => {
                let snapshot = Snapshot::deserialize(&artifact.data)?;
                Ok(ExecutableModule::new(InterpExecutable::new(snapshot)))
            },
            _ => Err(HoduError::UnsupportedOperation(
                format!(
                    "InterpRuntime only supports HoduSnapshot format, got {:?}",
                    artifact.format
                )
                .into(),
            )),
        }
    }

    fn load_file(&self, path: &Path, _device: Device) -> HoduResult<ExecutableModule> {
        let data = std::fs::read(path).map_err(|e| HoduError::IoError(format!("Failed to read file: {}", e)))?;
        let snapshot = Snapshot::deserialize(&data)?;
        Ok(ExecutableModule::new(InterpExecutable::new(snapshot)))
    }
}

impl InterpRuntime {
    /// Execute a Snapshot directly
    pub fn execute_snapshot(
        &self,
        snapshot: &Snapshot,
        inputs: &[(&str, &Tensor)],
    ) -> HoduResult<HashMap<String, Tensor>> {
        execute_snapshot(snapshot, inputs)
    }
}

/// Executable module for interpreter
pub struct InterpExecutable {
    snapshot: Snapshot,
}

impl InterpExecutable {
    pub fn new(snapshot: Snapshot) -> Self {
        Self { snapshot }
    }
}

impl ExecutableModuleInner for InterpExecutable {
    fn execute(&self, inputs: &[(&str, &Tensor)]) -> HoduResult<HashMap<String, Tensor>> {
        execute_snapshot(&self.snapshot, inputs)
    }
}

/// Execute a Snapshot with given inputs
fn execute_snapshot(snapshot: &Snapshot, inputs: &[(&str, &Tensor)]) -> HoduResult<HashMap<String, Tensor>> {
    let mut tensors: HashMap<SnapshotTensorId, Tensor> = HashMap::new();

    // 1. Load inputs
    for input_spec in &snapshot.inputs {
        let input_tensor = inputs
            .iter()
            .find(|(name, _)| *name == input_spec.name)
            .map(|(_, t)| *t)
            .ok_or_else(|| HoduError::InvalidArgument(format!("Missing input: {}", input_spec.name).into()))?;

        if input_tensor.shape() != input_spec.shape {
            return Err(HoduError::ShapeMismatch {
                expected: input_spec.shape.clone(),
                got: input_tensor.shape().clone(),
            });
        }
        if input_tensor.dtype() != input_spec.dtype {
            return Err(HoduError::DTypeMismatch {
                expected: input_spec.dtype,
                got: input_tensor.dtype(),
            });
        }

        tensors.insert(input_spec.id, input_tensor.clone());
    }

    // 2. Load constants
    for constant in &snapshot.constants {
        let tensor = tensor_from_constant(constant)?;
        tensors.insert(constant.id, tensor);
    }

    // 3. Execute nodes in order
    for node in &snapshot.nodes {
        let result = execute_node(node, &tensors)?;
        tensors.insert(node.output_id, result);
    }

    // 4. Collect outputs
    let mut outputs = HashMap::new();
    for target in &snapshot.targets {
        let tensor = tensors
            .get(&target.id)
            .ok_or_else(|| HoduError::InvalidArgument(format!("Output tensor not found: {}", target.name)))?;
        outputs.insert(target.name.clone(), tensor.clone());
    }

    Ok(outputs)
}

/// Create a Tensor from a SnapshotConstant
fn tensor_from_constant(constant: &SnapshotConstant) -> HoduResult<Tensor> {
    let shape = constant.shape.clone();
    let dtype = constant.dtype;
    let data = &constant.data;

    match dtype {
        DType::BOOL => {
            let values: Vec<bool> = data.iter().map(|&b| b != 0).collect();
            Tensor::from_slice(values, shape)
        },
        DType::F8E4M3 => {
            let values: Vec<F8E4M3> = data.iter().map(|&b| F8E4M3::from_bits(b)).collect();
            Tensor::from_slice(values, shape)
        },
        #[cfg(feature = "f8e5m2")]
        DType::F8E5M2 => {
            let values: Vec<F8E5M2> = data.iter().map(|&b| F8E5M2::from_bits(b)).collect();
            Tensor::from_slice(values, shape)
        },
        DType::BF16 => {
            let values: Vec<bf16> = data
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]))
                .collect();
            Tensor::from_slice(values, shape)
        },
        DType::F16 => {
            let values: Vec<f16> = data.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]])).collect();
            Tensor::from_slice(values, shape)
        },
        DType::F32 => {
            let values: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Tensor::from_slice(values, shape)
        },
        #[cfg(feature = "f64")]
        DType::F64 => {
            let values: Vec<f64> = data
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            Tensor::from_slice(values, shape)
        },
        DType::U8 => {
            let values: Vec<u8> = data.to_vec();
            Tensor::from_slice(values, shape)
        },
        #[cfg(feature = "u16")]
        DType::U16 => {
            let values: Vec<u16> = data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            Tensor::from_slice(values, shape)
        },
        DType::U32 => {
            let values: Vec<u32> = data
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Tensor::from_slice(values, shape)
        },
        #[cfg(feature = "u64")]
        DType::U64 => {
            let values: Vec<u64> = data
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            Tensor::from_slice(values, shape)
        },
        DType::I8 => {
            let values: Vec<i8> = data.iter().map(|&b| b as i8).collect();
            Tensor::from_slice(values, shape)
        },
        #[cfg(feature = "i16")]
        DType::I16 => {
            let values: Vec<i16> = data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
            Tensor::from_slice(values, shape)
        },
        DType::I32 => {
            let values: Vec<i32> = data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Tensor::from_slice(values, shape)
        },
        #[cfg(feature = "i64")]
        DType::I64 => {
            let values: Vec<i64> = data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            Tensor::from_slice(values, shape)
        },
        #[allow(unreachable_patterns)]
        _ => Err(HoduError::UnsupportedDType {
            dtype,
            reason: "Constant loading not implemented for this dtype".into(),
        }),
    }
}

/// Execute a single node
fn execute_node(node: &SnapshotNode, tensors: &HashMap<SnapshotTensorId, Tensor>) -> HoduResult<Tensor> {
    let get_input = |idx: usize| -> HoduResult<&Tensor> {
        let id = node
            .input_ids
            .get(idx)
            .ok_or_else(|| HoduError::InvalidArgument(format!("Missing input {} for op {:?}", idx, node.op).into()))?;
        tensors
            .get(id)
            .ok_or_else(|| HoduError::InvalidArgument(format!("Tensor {:?} not found", id).into()))
    };

    let get_inputs = || -> HoduResult<Vec<&Tensor>> {
        node.input_ids
            .iter()
            .map(|id| {
                tensors
                    .get(id)
                    .ok_or_else(|| HoduError::InvalidArgument(format!("Tensor {:?} not found", id).into()))
            })
            .collect()
    };

    match &node.op {
        // ==================== Binary ====================
        Op::Binary(BinaryOp::Add) => get_input(0)?.add(get_input(1)?),
        Op::Binary(BinaryOp::Sub) => get_input(0)?.sub(get_input(1)?),
        Op::Binary(BinaryOp::Mul) => get_input(0)?.mul(get_input(1)?),
        Op::Binary(BinaryOp::Div) => get_input(0)?.div(get_input(1)?),
        Op::Binary(BinaryOp::Pow) => get_input(0)?.pow(get_input(1)?),
        Op::Binary(BinaryOp::Maximum) => get_input(0)?.maximum(get_input(1)?),
        Op::Binary(BinaryOp::Minimum) => get_input(0)?.minimum(get_input(1)?),

        // ==================== BinaryLogical ====================
        Op::BinaryLogical(BinaryLogicalOp::LogicalAnd) => get_input(0)?.logical_and(get_input(1)?),
        Op::BinaryLogical(BinaryLogicalOp::LogicalOr) => get_input(0)?.logical_or(get_input(1)?),
        Op::BinaryLogical(BinaryLogicalOp::LogicalXor) => get_input(0)?.logical_xor(get_input(1)?),

        // ==================== Cmp ====================
        Op::Cmp(CmpOp::Eq) => get_input(0)?.eq(get_input(1)?),
        Op::Cmp(CmpOp::Ne) => get_input(0)?.ne(get_input(1)?),
        Op::Cmp(CmpOp::Lt) => get_input(0)?.lt(get_input(1)?),
        Op::Cmp(CmpOp::Le) => get_input(0)?.le(get_input(1)?),
        Op::Cmp(CmpOp::Gt) => get_input(0)?.gt(get_input(1)?),
        Op::Cmp(CmpOp::Ge) => get_input(0)?.ge(get_input(1)?),

        // ==================== CmpScalar ====================
        Op::CmpScalar(CmpScalarOp::EqScalar) => {
            let scalar = get_cmp_scalar_param(node)?;
            get_input(0)?.eq_scalar(scalar)
        },
        Op::CmpScalar(CmpScalarOp::NeScalar) => {
            let scalar = get_cmp_scalar_param(node)?;
            get_input(0)?.ne_scalar(scalar)
        },
        Op::CmpScalar(CmpScalarOp::LtScalar) => {
            let scalar = get_cmp_scalar_param(node)?;
            get_input(0)?.lt_scalar(scalar)
        },
        Op::CmpScalar(CmpScalarOp::LeScalar) => {
            let scalar = get_cmp_scalar_param(node)?;
            get_input(0)?.le_scalar(scalar)
        },
        Op::CmpScalar(CmpScalarOp::GtScalar) => {
            let scalar = get_cmp_scalar_param(node)?;
            get_input(0)?.gt_scalar(scalar)
        },
        Op::CmpScalar(CmpScalarOp::GeScalar) => {
            let scalar = get_cmp_scalar_param(node)?;
            get_input(0)?.ge_scalar(scalar)
        },

        // ==================== Unary ====================
        Op::Unary(UnaryOp::Neg) => get_input(0)?.neg(),
        Op::Unary(UnaryOp::Abs) => get_input(0)?.abs(),
        Op::Unary(UnaryOp::Sign) => get_input(0)?.sign(),
        Op::Unary(UnaryOp::Square) => get_input(0)?.square(),
        Op::Unary(UnaryOp::Sqrt) => get_input(0)?.sqrt(),
        Op::Unary(UnaryOp::Recip) => get_input(0)?.recip(),
        Op::Unary(UnaryOp::Relu) => get_input(0)?.relu(),
        Op::Unary(UnaryOp::Sigmoid) => get_input(0)?.sigmoid(),
        Op::Unary(UnaryOp::Tanh) => get_input(0)?.tanh(),
        Op::Unary(UnaryOp::Gelu) => get_input(0)?.gelu(),
        Op::Unary(UnaryOp::Softplus) => get_input(0)?.softplus(),
        Op::Unary(UnaryOp::Silu) => get_input(0)?.silu(),
        Op::Unary(UnaryOp::Mish) => get_input(0)?.mish(),
        Op::Unary(UnaryOp::Sin) => get_input(0)?.sin(),
        Op::Unary(UnaryOp::Cos) => get_input(0)?.cos(),
        Op::Unary(UnaryOp::Tan) => get_input(0)?.tan(),
        Op::Unary(UnaryOp::Exp) => get_input(0)?.exp(),
        Op::Unary(UnaryOp::Exp2) => get_input(0)?.exp2(),
        Op::Unary(UnaryOp::Exp10) => get_input(0)?.exp10(),
        Op::Unary(UnaryOp::Ln) => get_input(0)?.ln(),
        Op::Unary(UnaryOp::Log2) => get_input(0)?.log2(),
        Op::Unary(UnaryOp::Log10) => get_input(0)?.log10(),

        // ==================== UnaryLogical ====================
        Op::UnaryLogical(UnaryLogicalOp::LogicalNot) => get_input(0)?.logical_not(),

        // ==================== UnaryScalar ====================
        Op::UnaryScalar(UnaryScalarOp::AddScalar) => {
            let scalar = get_unary_scalar_param(node)?;
            get_input(0)?.add_scalar(scalar)
        },
        Op::UnaryScalar(UnaryScalarOp::SubScalar) => {
            let scalar = get_unary_scalar_param(node)?;
            get_input(0)?.sub_scalar(scalar)
        },
        Op::UnaryScalar(UnaryScalarOp::MulScalar) => {
            let scalar = get_unary_scalar_param(node)?;
            get_input(0)?.mul_scalar(scalar)
        },
        Op::UnaryScalar(UnaryScalarOp::DivScalar) => {
            let scalar = get_unary_scalar_param(node)?;
            get_input(0)?.div_scalar(scalar)
        },
        Op::UnaryScalar(UnaryScalarOp::PowScalar) => {
            let scalar = get_unary_scalar_param(node)?;
            get_input(0)?.pow_scalar(scalar)
        },
        Op::UnaryScalar(UnaryScalarOp::MaximumScalar) => {
            let scalar = get_unary_scalar_param(node)?;
            get_input(0)?.maximum_scalar(scalar)
        },
        Op::UnaryScalar(UnaryScalarOp::MinimumScalar) => {
            let scalar = get_unary_scalar_param(node)?;
            get_input(0)?.minimum_scalar(scalar)
        },
        Op::UnaryScalar(UnaryScalarOp::LeakyRelu) => {
            let scalar = get_unary_scalar_param(node)?;
            get_input(0)?.leaky_relu(scalar)
        },
        Op::UnaryScalar(UnaryScalarOp::Elu) => {
            let scalar = get_unary_scalar_param(node)?;
            get_input(0)?.elu(scalar)
        },
        Op::UnaryScalar(UnaryScalarOp::Prelu) => {
            let scalar = get_unary_scalar_param(node)?;
            get_input(0)?.prelu(scalar)
        },

        // ==================== Matrix ====================
        Op::Matrix(MatrixOp::Matmul) => get_input(0)?.matmul(get_input(1)?),
        Op::Matrix(MatrixOp::Dot) => get_input(0)?.dot(get_input(1)?),

        // ==================== Reduce ====================
        Op::Reduce(reduce_op) => {
            let (dims, keep_dim) = get_reduce_params(node)?;
            let input = get_input(0)?;
            match reduce_op {
                ReduceOp::Sum => input.sum(&dims, keep_dim),
                ReduceOp::Mean => input.mean(&dims, keep_dim),
                ReduceOp::Max => input.max(&dims, keep_dim),
                ReduceOp::Min => input.min(&dims, keep_dim),
                ReduceOp::Prod => input.prod(&dims, keep_dim),
                ReduceOp::Std => input.std(&dims, keep_dim),
                ReduceOp::Var => input.var(&dims, keep_dim),
                ReduceOp::Norm => input.l2_norm(&dims, keep_dim),
                ReduceOp::ArgMax => input.argmax(&dims, keep_dim),
                ReduceOp::ArgMin => input.argmin(&dims, keep_dim),
                ReduceOp::Any => input.any(&dims, keep_dim),
                ReduceOp::All => input.all(&dims, keep_dim),
            }
        },

        // ==================== Concat ====================
        Op::Concat(ConcatOp::Concat) => {
            let dim = get_concat_param(node)?;
            let inputs = get_inputs()?;
            Tensor::concat(&inputs, dim)
        },

        // ==================== Split ====================
        Op::Split(SplitOp::Split) => {
            let (dim, sizes, output_index) = get_split_params(node)?;
            let input = get_input(0)?;
            let results = input.split(&sizes, dim)?;
            results.into_iter().nth(output_index).ok_or_else(|| {
                HoduError::InvalidArgument(format!("Split output_index {} out of bounds", output_index).into())
            })
        },

        // ==================== Indexing ====================
        Op::Indexing(IndexingOp::IndexSelect) => {
            let dim = get_index_select_param(node)?;
            get_input(0)?.index_select(dim, get_input(1)?)
        },
        Op::Indexing(IndexingOp::IndexPut) => {
            let dim = get_index_put_param(node)?;
            get_input(0)?.index_put(dim, get_input(1)?, get_input(2)?)
        },
        Op::Indexing(IndexingOp::Gather) => {
            let dim = get_gather_param(node)?;
            get_input(0)?.gather(dim, get_input(1)?)
        },
        Op::Indexing(IndexingOp::Scatter) => {
            let dim = get_scatter_param(node)?;
            get_input(0)?.scatter(dim, get_input(1)?, get_input(2)?)
        },
        Op::Indexing(IndexingOp::ScatterAdd) => {
            let dim = get_scatter_add_param(node)?;
            get_input(0)?.scatter_add(dim, get_input(1)?, get_input(2)?)
        },
        Op::Indexing(IndexingOp::ScatterMax) => {
            let dim = get_scatter_max_param(node)?;
            get_input(0)?.scatter_max(dim, get_input(1)?, get_input(2)?)
        },
        Op::Indexing(IndexingOp::ScatterMin) => {
            let dim = get_scatter_min_param(node)?;
            get_input(0)?.scatter_min(dim, get_input(1)?, get_input(2)?)
        },

        // ==================== Conv ====================
        Op::Conv(ConvOp::Conv1d) => {
            let params = get_conv1d_params(node)?;
            get_input(0)?.conv1d(get_input(1)?, params.padding, params.stride, params.dilation)
        },
        Op::Conv(ConvOp::Conv2d) => {
            let params = get_conv2d_params(node)?;
            get_input(0)?.conv2d(get_input(1)?, params.padding, params.stride, params.dilation)
        },
        Op::Conv(ConvOp::Conv3d) => {
            let params = get_conv3d_params(node)?;
            get_input(0)?.conv3d(get_input(1)?, params.padding, params.stride, params.dilation)
        },
        Op::Conv(ConvOp::ConvTranspose1d) => {
            let params = get_conv_transpose1d_params(node)?;
            get_input(0)?.conv_transpose1d(
                get_input(1)?,
                params.padding,
                params.output_padding,
                params.stride,
                params.dilation,
            )
        },
        Op::Conv(ConvOp::ConvTranspose2d) => {
            let params = get_conv_transpose2d_params(node)?;
            get_input(0)?.conv_transpose2d(
                get_input(1)?,
                params.padding,
                params.output_padding,
                params.stride,
                params.dilation,
            )
        },
        Op::Conv(ConvOp::ConvTranspose3d) => {
            let params = get_conv_transpose3d_params(node)?;
            get_input(0)?.conv_transpose3d(
                get_input(1)?,
                params.padding,
                params.output_padding,
                params.stride,
                params.dilation,
            )
        },
        Op::Conv(ConvOp::Conv1dGradWeight) => {
            let params = get_conv1d_grad_weight_params(node)?;
            let weight_shape = [params.out_channels, params.in_channels, params.kernel_size];
            get_input(0)?.conv1d_grad_weight(
                get_input(1)?,
                &weight_shape,
                params.stride,
                params.padding,
                params.dilation,
            )
        },
        Op::Conv(ConvOp::Conv2dGradWeight) => {
            let params = get_conv2d_grad_weight_params(node)?;
            let weight_shape = [
                params.out_channels,
                params.in_channels,
                params.kernel_height,
                params.kernel_width,
            ];
            get_input(0)?.conv2d_grad_weight(
                get_input(1)?,
                &weight_shape,
                params.stride,
                params.padding,
                params.dilation,
            )
        },
        Op::Conv(ConvOp::Conv3dGradWeight) => {
            let params = get_conv3d_grad_weight_params(node)?;
            let weight_shape = [
                params.out_channels,
                params.in_channels,
                params.kernel_depth,
                params.kernel_height,
                params.kernel_width,
            ];
            get_input(0)?.conv3d_grad_weight(
                get_input(1)?,
                &weight_shape,
                params.stride,
                params.padding,
                params.dilation,
            )
        },
        Op::Conv(ConvOp::ConvTranspose1dGradWeight) => {
            let params = get_conv_transpose1d_grad_weight_params(node)?;
            let weight_shape = [params.in_channels, params.out_channels, params.kernel_size];
            get_input(0)?.conv_transpose1d_grad_weight(
                get_input(1)?,
                &weight_shape,
                params.stride,
                params.padding,
                params.dilation,
            )
        },
        Op::Conv(ConvOp::ConvTranspose2dGradWeight) => {
            let params = get_conv_transpose2d_grad_weight_params(node)?;
            let weight_shape = [
                params.in_channels,
                params.out_channels,
                params.kernel_height,
                params.kernel_width,
            ];
            get_input(0)?.conv_transpose2d_grad_weight(
                get_input(1)?,
                &weight_shape,
                params.stride,
                params.padding,
                params.dilation,
            )
        },
        Op::Conv(ConvOp::ConvTranspose3dGradWeight) => {
            let params = get_conv_transpose3d_grad_weight_params(node)?;
            let weight_shape = [
                params.in_channels,
                params.out_channels,
                params.kernel_depth,
                params.kernel_height,
                params.kernel_width,
            ];
            get_input(0)?.conv_transpose3d_grad_weight(
                get_input(1)?,
                &weight_shape,
                params.stride,
                params.padding,
                params.dilation,
            )
        },

        // ==================== Windowing ====================
        Op::Windowing(WindowingOp::ReduceWindowMax) => {
            let params = get_reduce_window_params(node)?;
            get_input(0)?.reduce_window(&params.window_shape, &params.strides, &params.padding, "max")
        },
        Op::Windowing(WindowingOp::ReduceWindowMean) => {
            let params = get_reduce_window_params(node)?;
            get_input(0)?.reduce_window(&params.window_shape, &params.strides, &params.padding, "mean")
        },
        Op::Windowing(WindowingOp::ReduceWindowSum) => {
            let params = get_reduce_window_params(node)?;
            get_input(0)?.reduce_window(&params.window_shape, &params.strides, &params.padding, "sum")
        },
        Op::Windowing(WindowingOp::ReduceWindowMin) => {
            let params = get_reduce_window_params(node)?;
            get_input(0)?.reduce_window(&params.window_shape, &params.strides, &params.padding, "min")
        },

        // ==================== Shape ====================
        Op::Shape(ShapeOp::Reshape) => {
            let target_shape = node.output_layout.shape();
            get_input(0)?.reshape(target_shape)
        },
        Op::Shape(ShapeOp::Flatten) => get_input(0)?.flatten(),
        Op::Shape(ShapeOp::Squeeze) => {
            let dims = get_squeeze_dims(node)?;
            get_input(0)?.squeeze(&dims)
        },
        Op::Shape(ShapeOp::Unsqueeze) => {
            let dim = get_unsqueeze_dim(node)?;
            get_input(0)?.unsqueeze(dim)
        },
        Op::Shape(ShapeOp::Broadcast) => {
            let target_shape = node.output_layout.shape();
            get_input(0)?.broadcast(target_shape)
        },
        Op::Shape(ShapeOp::Transpose) => {
            let (dim1, dim2) = get_transpose_dims(node)?;
            get_input(0)?.transpose(dim1, dim2)
        },
        Op::Shape(ShapeOp::Permute) => {
            let axes = get_permute_axes(node)?;
            get_input(0)?.permute(&axes)
        },

        // ==================== ShapeScalars ====================
        Op::ShapeScalars(ShapeScalarsOp::Slice) => {
            let params = get_slice_params(node)?;
            let end = if params.end.to_i32() == i32::MAX {
                None
            } else {
                Some(params.end)
            };
            get_input(0)?.slice(params.dim, params.start, end, params.step)
        },

        // ==================== Cast ====================
        Op::Cast(CastOp::ToDType) => {
            let target_dtype = get_to_dtype_param(node)?;
            get_input(0)?.to_dtype(target_dtype)
        },

        // ==================== Memory ====================
        Op::Memory(MemoryOp::Contiguous) => get_input(0)?.contiguous(),

        // ==================== Dummy ====================
        Op::Dummy => Err(HoduError::UnsupportedOperation("Dummy op cannot be executed".into())),
    }
}

// ==================== Parameter extraction helpers ====================

fn get_unary_scalar_param(node: &SnapshotNode) -> HoduResult<hodu_core::scalar::Scalar> {
    match &node.params {
        Some(OpParams::UnaryScalar(p)) => Ok(p.scalar),
        _ => Err(HoduError::InvalidArgument("Expected UnaryScalar params".into())),
    }
}

fn get_cmp_scalar_param(node: &SnapshotNode) -> HoduResult<hodu_core::scalar::Scalar> {
    match &node.params {
        Some(OpParams::CmpScalar(p)) => Ok(p.scalar),
        _ => Err(HoduError::InvalidArgument("Expected CmpScalar params".into())),
    }
}

fn get_reduce_params(node: &SnapshotNode) -> HoduResult<(Vec<usize>, bool)> {
    match &node.params {
        Some(OpParams::Reduce(p)) => {
            let dims: Vec<usize> = p.dims.iter().map(|s| s.to_usize()).collect();
            Ok((dims, p.keep_dim))
        },
        _ => Err(HoduError::InvalidArgument("Expected Reduce params".into())),
    }
}

fn get_concat_param(node: &SnapshotNode) -> HoduResult<i32> {
    match &node.params {
        Some(OpParams::Concat(p)) => Ok(p.dim.to_i32()),
        _ => Err(HoduError::InvalidArgument("Expected Concat params".into())),
    }
}

fn get_split_params(node: &SnapshotNode) -> HoduResult<(i32, Vec<usize>, usize)> {
    match &node.params {
        Some(OpParams::Split(p)) => {
            let sizes: Vec<usize> = p.sizes.iter().map(|s| s.to_usize()).collect();
            Ok((p.dim.to_i32(), sizes, p.output_index))
        },
        _ => Err(HoduError::InvalidArgument("Expected Split params".into())),
    }
}

fn get_index_select_param(node: &SnapshotNode) -> HoduResult<i32> {
    match &node.params {
        Some(OpParams::IndexSelect(p)) => Ok(p.dim.to_i32()),
        _ => Err(HoduError::InvalidArgument("Expected IndexSelect params".into())),
    }
}

fn get_index_put_param(node: &SnapshotNode) -> HoduResult<i32> {
    match &node.params {
        Some(OpParams::IndexPut(p)) => Ok(p.dim.to_i32()),
        _ => Err(HoduError::InvalidArgument("Expected IndexPut params".into())),
    }
}

fn get_gather_param(node: &SnapshotNode) -> HoduResult<i32> {
    match &node.params {
        Some(OpParams::Gather(p)) => Ok(p.dim.to_i32()),
        _ => Err(HoduError::InvalidArgument("Expected Gather params".into())),
    }
}

fn get_scatter_param(node: &SnapshotNode) -> HoduResult<i32> {
    match &node.params {
        Some(OpParams::Scatter(p)) => Ok(p.dim.to_i32()),
        _ => Err(HoduError::InvalidArgument("Expected Scatter params".into())),
    }
}

fn get_scatter_add_param(node: &SnapshotNode) -> HoduResult<i32> {
    match &node.params {
        Some(OpParams::ScatterAdd(p)) => Ok(p.dim.to_i32()),
        _ => Err(HoduError::InvalidArgument("Expected ScatterAdd params".into())),
    }
}

fn get_scatter_max_param(node: &SnapshotNode) -> HoduResult<i32> {
    match &node.params {
        Some(OpParams::ScatterMax(p)) => Ok(p.dim.to_i32()),
        _ => Err(HoduError::InvalidArgument("Expected ScatterMax params".into())),
    }
}

fn get_scatter_min_param(node: &SnapshotNode) -> HoduResult<i32> {
    match &node.params {
        Some(OpParams::ScatterMin(p)) => Ok(p.dim.to_i32()),
        _ => Err(HoduError::InvalidArgument("Expected ScatterMin params".into())),
    }
}

fn get_conv1d_params(node: &SnapshotNode) -> HoduResult<&Conv1dParams> {
    match &node.params {
        Some(OpParams::Conv1d(p)) => Ok(p),
        _ => Err(HoduError::InvalidArgument("Expected Conv1d params".into())),
    }
}

fn get_conv2d_params(node: &SnapshotNode) -> HoduResult<&Conv2dParams> {
    match &node.params {
        Some(OpParams::Conv2d(p)) => Ok(p),
        _ => Err(HoduError::InvalidArgument("Expected Conv2d params".into())),
    }
}

fn get_conv3d_params(node: &SnapshotNode) -> HoduResult<&Conv3dParams> {
    match &node.params {
        Some(OpParams::Conv3d(p)) => Ok(p),
        _ => Err(HoduError::InvalidArgument("Expected Conv3d params".into())),
    }
}

fn get_conv_transpose1d_params(node: &SnapshotNode) -> HoduResult<&ConvTranspose1dParams> {
    match &node.params {
        Some(OpParams::ConvTranspose1d(p)) => Ok(p),
        _ => Err(HoduError::InvalidArgument("Expected ConvTranspose1d params".into())),
    }
}

fn get_conv_transpose2d_params(node: &SnapshotNode) -> HoduResult<&ConvTranspose2dParams> {
    match &node.params {
        Some(OpParams::ConvTranspose2d(p)) => Ok(p),
        _ => Err(HoduError::InvalidArgument("Expected ConvTranspose2d params".into())),
    }
}

fn get_conv_transpose3d_params(node: &SnapshotNode) -> HoduResult<&ConvTranspose3dParams> {
    match &node.params {
        Some(OpParams::ConvTranspose3d(p)) => Ok(p),
        _ => Err(HoduError::InvalidArgument("Expected ConvTranspose3d params".into())),
    }
}

fn get_conv1d_grad_weight_params(node: &SnapshotNode) -> HoduResult<&Conv1dGradWeightParams> {
    match &node.params {
        Some(OpParams::Conv1dGradWeight(p)) => Ok(p),
        _ => Err(HoduError::InvalidArgument("Expected Conv1dGradWeight params".into())),
    }
}

fn get_conv2d_grad_weight_params(node: &SnapshotNode) -> HoduResult<&Conv2dGradWeightParams> {
    match &node.params {
        Some(OpParams::Conv2dGradWeight(p)) => Ok(p),
        _ => Err(HoduError::InvalidArgument("Expected Conv2dGradWeight params".into())),
    }
}

fn get_conv3d_grad_weight_params(node: &SnapshotNode) -> HoduResult<&Conv3dGradWeightParams> {
    match &node.params {
        Some(OpParams::Conv3dGradWeight(p)) => Ok(p),
        _ => Err(HoduError::InvalidArgument("Expected Conv3dGradWeight params".into())),
    }
}

fn get_conv_transpose1d_grad_weight_params(node: &SnapshotNode) -> HoduResult<&ConvTranspose1dGradWeightParams> {
    match &node.params {
        Some(OpParams::ConvTranspose1dGradWeight(p)) => Ok(p),
        _ => Err(HoduError::InvalidArgument(
            "Expected ConvTranspose1dGradWeight params".into(),
        )),
    }
}

fn get_conv_transpose2d_grad_weight_params(node: &SnapshotNode) -> HoduResult<&ConvTranspose2dGradWeightParams> {
    match &node.params {
        Some(OpParams::ConvTranspose2dGradWeight(p)) => Ok(p),
        _ => Err(HoduError::InvalidArgument(
            "Expected ConvTranspose2dGradWeight params".into(),
        )),
    }
}

fn get_conv_transpose3d_grad_weight_params(node: &SnapshotNode) -> HoduResult<&ConvTranspose3dGradWeightParams> {
    match &node.params {
        Some(OpParams::ConvTranspose3dGradWeight(p)) => Ok(p),
        _ => Err(HoduError::InvalidArgument(
            "Expected ConvTranspose3dGradWeight params".into(),
        )),
    }
}

fn get_reduce_window_params(node: &SnapshotNode) -> HoduResult<&ReduceWindowParams> {
    match &node.params {
        Some(OpParams::ReduceWindow(p)) => Ok(p),
        _ => Err(HoduError::InvalidArgument("Expected ReduceWindow params".into())),
    }
}

fn get_squeeze_dims(node: &SnapshotNode) -> HoduResult<Vec<i32>> {
    // Squeeze dims are derived from input_layout vs output_layout
    let input_shape = node
        .input_layouts
        .first()
        .map(|l| l.shape())
        .ok_or_else(|| HoduError::InvalidArgument("Squeeze requires input layout".into()))?;
    let output_shape = node.output_layout.shape();

    let input_dims = input_shape.dims();
    let output_dims = output_shape.dims();

    let mut squeeze_dims = Vec::new();
    let mut out_idx = 0;

    for (in_idx, &dim) in input_dims.iter().enumerate() {
        if out_idx < output_dims.len() && dim == output_dims[out_idx] {
            out_idx += 1;
        } else if dim == 1 {
            squeeze_dims.push(in_idx as i32);
        }
    }

    Ok(squeeze_dims)
}

fn get_unsqueeze_dim(node: &SnapshotNode) -> HoduResult<i32> {
    // Unsqueeze dim is derived from input_layout vs output_layout
    let input_shape = node
        .input_layouts
        .first()
        .map(|l| l.shape())
        .ok_or_else(|| HoduError::InvalidArgument("Unsqueeze requires input layout".into()))?;
    let output_shape = node.output_layout.shape();

    let input_dims = input_shape.dims();
    let output_dims = output_shape.dims();

    for (idx, &dim) in output_dims.iter().enumerate() {
        if dim == 1 {
            if idx >= input_dims.len() || input_dims[idx] != 1 {
                return Ok(idx as i32);
            }
        }
    }

    Ok(0)
}

fn get_transpose_dims(node: &SnapshotNode) -> HoduResult<(i32, i32)> {
    // Default transpose swaps last two dimensions
    let input_shape = node
        .input_layouts
        .first()
        .map(|l| l.shape())
        .ok_or_else(|| HoduError::InvalidArgument("Transpose requires input layout".into()))?;
    let ndim = input_shape.ndim() as i32;
    Ok((ndim - 2, ndim - 1))
}

fn get_permute_axes(node: &SnapshotNode) -> HoduResult<Vec<i32>> {
    // Derive permute axes from input/output strides
    let input_layout = node
        .input_layouts
        .first()
        .ok_or_else(|| HoduError::InvalidArgument("Permute requires input layout".into()))?;
    let output_layout = &node.output_layout;

    let input_strides = input_layout.strides();
    let output_strides = output_layout.strides();
    let output_shape = output_layout.shape();

    let mut axes = Vec::with_capacity(output_strides.len());

    for (out_idx, &out_stride) in output_strides.iter().enumerate() {
        let out_dim = output_shape.dims()[out_idx];
        for (in_idx, &in_stride) in input_strides.iter().enumerate() {
            if in_stride == out_stride && !axes.contains(&(in_idx as i32)) {
                // Also check dimension size matches
                let in_dim = input_layout.shape().dims()[in_idx];
                if in_dim == out_dim {
                    axes.push(in_idx as i32);
                    break;
                }
            }
        }
    }

    if axes.len() != output_strides.len() {
        // Fallback: identity permutation
        axes = (0..output_strides.len() as i32).collect();
    }

    Ok(axes)
}

fn get_slice_params(node: &SnapshotNode) -> HoduResult<&SliceParams> {
    match &node.params {
        Some(OpParams::Slice(p)) => Ok(p),
        _ => Err(HoduError::InvalidArgument("Expected Slice params".into())),
    }
}

fn get_to_dtype_param(node: &SnapshotNode) -> HoduResult<DType> {
    match &node.params {
        Some(OpParams::ToDType(p)) => Ok(p.dtype),
        _ => Ok(node.output_dtype),
    }
}
