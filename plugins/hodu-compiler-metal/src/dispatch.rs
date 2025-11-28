//! Dispatch manifest for Metal kernel execution
//!
//! Converts Snapshot nodes to a sequence of kernel dispatches.

use hodu_core::ops::Op;
use hodu_core::script::Snapshot;
use hodu_core::types::DType;
use serde::{Deserialize, Serialize};

/// Dispatch manifest for executing a compiled graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchManifest {
    /// Model name
    pub name: Option<String>,
    /// Input tensor specifications
    pub inputs: Vec<TensorSpec>,
    /// Output tensor specifications
    pub outputs: Vec<TensorSpec>,
    /// Constant tensor data (embedded)
    pub constants: Vec<ConstantData>,
    /// Sequence of kernel dispatches
    pub dispatches: Vec<KernelDispatch>,
    /// Total number of intermediate buffers needed
    pub num_buffers: usize,
}

/// Tensor specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub buffer_id: usize,
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// Constant tensor data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantData {
    pub buffer_id: usize,
    pub shape: Vec<usize>,
    pub dtype: String,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

/// A single kernel dispatch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelDispatch {
    /// Kernel function name (e.g., "hodu_metal_add_f32")
    pub kernel_name: String,
    /// Input buffer IDs
    pub input_buffers: Vec<usize>,
    /// Output buffer ID
    pub output_buffer: usize,
    /// Metadata for the kernel
    pub metadata: Vec<usize>,
    /// Thread grid size
    pub grid_size: usize,
    /// Optional scalar value (for scalar ops)
    pub scalar: Option<ScalarValue>,
}

/// Scalar value for scalar operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalarValue {
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    U32(u32),
    U64(u64),
    Bool(bool),
}

impl DispatchManifest {
    /// Generate dispatch manifest from a Snapshot
    pub fn from_snapshot(snapshot: &Snapshot) -> Self {
        let mut dispatches = Vec::new();
        let mut num_buffers = 0;

        // Map snapshot tensor IDs to buffer IDs
        let mut tensor_to_buffer: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

        // Allocate buffers for inputs
        let inputs: Vec<TensorSpec> = snapshot.inputs.iter().map(|input| {
            let buffer_id = num_buffers;
            tensor_to_buffer.insert(input.id.0, buffer_id);
            num_buffers += 1;
            TensorSpec {
                name: input.name.clone(),
                buffer_id,
                shape: input.shape.dims().to_vec(),
                dtype: dtype_to_string(input.dtype),
            }
        }).collect();

        // Allocate buffers for constants
        let constants: Vec<ConstantData> = snapshot.constants.iter().map(|constant| {
            let buffer_id = num_buffers;
            tensor_to_buffer.insert(constant.id.0, buffer_id);
            num_buffers += 1;
            ConstantData {
                buffer_id,
                shape: constant.shape.dims().to_vec(),
                dtype: dtype_to_string(constant.dtype),
                data: constant.data.clone(),
            }
        }).collect();

        // Generate dispatches for each node
        for node in &snapshot.nodes {
            let output_buffer = num_buffers;
            tensor_to_buffer.insert(node.output_id.0, output_buffer);
            num_buffers += 1;

            let input_buffers: Vec<usize> = node.input_ids.iter()
                .map(|id| tensor_to_buffer.get(&id.0).copied().unwrap_or(0))
                .collect();

            let kernel_name = op_to_kernel_name(&node.op, node.output_dtype);
            let grid_size = node.output_layout.size();

            // Build metadata
            let metadata = build_metadata(&node.input_layouts, &node.output_layout);

            dispatches.push(KernelDispatch {
                kernel_name,
                input_buffers,
                output_buffer,
                metadata,
                grid_size,
                scalar: None, // TODO: extract from OpParams
            });
        }

        // Build a map of tensor_id -> (shape, dtype) from nodes
        let mut tensor_info: std::collections::HashMap<usize, (Vec<usize>, DType)> = std::collections::HashMap::new();
        for node in &snapshot.nodes {
            tensor_info.insert(
                node.output_id.0,
                (node.output_layout.shape().dims().to_vec(), node.output_dtype),
            );
        }

        // Map outputs
        let outputs: Vec<TensorSpec> = snapshot.targets.iter().map(|target| {
            let buffer_id = tensor_to_buffer.get(&target.id.0).copied().unwrap_or(0);
            let (shape, dtype) = tensor_info.get(&target.id.0)
                .cloned()
                .unwrap_or_else(|| (Vec::new(), DType::F32));
            TensorSpec {
                name: target.name.clone(),
                buffer_id,
                shape,
                dtype: dtype_to_string(dtype),
            }
        }).collect();

        DispatchManifest {
            name: snapshot.name.clone(),
            inputs,
            outputs,
            constants,
            dispatches,
            num_buffers,
        }
    }

    /// Serialize to JSON bytes
    pub fn to_json(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Deserialize from JSON bytes
    pub fn from_json(data: &[u8]) -> Option<Self> {
        serde_json::from_slice(data).ok()
    }
}

fn dtype_to_string(dtype: DType) -> String {
    format!("{}", dtype)
}

fn op_to_kernel_name(op: &Op, dtype: DType) -> String {
    let dtype_suffix = dtype_to_string(dtype);

    match op {
        Op::Binary(binary_op) => format!("hodu_metal_{}_{}", binary_op, dtype_suffix),
        Op::BinaryLogical(op) => format!("hodu_metal_{}_{}", op, dtype_suffix),
        Op::Cmp(op) => format!("hodu_metal_{}_{}", op, dtype_suffix),
        Op::CmpScalar(op) => format!("hodu_metal_{}_{}", op, dtype_suffix),
        Op::Unary(op) => format!("hodu_metal_{}_{}", op, dtype_suffix),
        Op::UnaryLogical(op) => format!("hodu_metal_{}_{}", op, dtype_suffix),
        Op::UnaryScalar(op) => format!("hodu_metal_{}_{}", op, dtype_suffix),
        Op::Matrix(op) => format!("hodu_metal_{}_{}", op, dtype_suffix),
        Op::Reduce(op) => format!("hodu_metal_{}_{}", op, dtype_suffix),
        Op::Concat(op) => format!("hodu_metal_{}_{}", op, dtype_suffix),
        Op::Split(op) => format!("hodu_metal_{}_{}", op, dtype_suffix),
        Op::Indexing(op) => format!("hodu_metal_{}_{}", op, dtype_suffix),
        Op::Conv(op) => format!("hodu_metal_{}_{}", op, dtype_suffix),
        Op::Windowing(op) => format!("hodu_metal_{}_{}", op, dtype_suffix),
        Op::Shape(_) => "noop".to_string(), // Shape ops don't need kernels
        Op::ShapeScalars(_) => "noop".to_string(),
        Op::Cast(_) => format!("hodu_metal_cast_{}", dtype_suffix), // TODO: need src dtype too
        Op::Memory(op) => format!("hodu_metal_{}_{}", op, dtype_suffix),
        Op::Dummy => "noop".to_string(),
    }
}

fn build_metadata(input_layouts: &[hodu_core::types::Layout], output_layout: &hodu_core::types::Layout) -> Vec<usize> {
    // Basic metadata: [num_els, num_dims, dims..., strides..., offset]
    let shape = output_layout.shape();
    let strides = output_layout.strides();
    let offset = output_layout.offset();
    let num_els = output_layout.size();
    let num_dims = shape.ndim();

    let mut metadata = Vec::with_capacity(2 + num_dims * 2 + 1);
    metadata.push(num_els);
    metadata.push(num_dims);
    for i in 0..num_dims {
        metadata.push(shape[i]);
    }
    for &stride in strides.iter().take(num_dims) {
        metadata.push(stride);
    }
    metadata.push(offset);

    // TODO: Add input layouts for binary ops, etc.
    let _ = input_layouts;

    metadata
}
