//! Dispatch manifest for Metal kernel execution
//!
//! Converts Snapshot nodes to a sequence of kernel dispatches.

use hodu_core::{op_params::OpParams, ops::Op, script::Snapshot, types::DType};
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
        let inputs: Vec<TensorSpec> = snapshot
            .inputs
            .iter()
            .map(|input| {
                let buffer_id = num_buffers;
                tensor_to_buffer.insert(input.id.0, buffer_id);
                num_buffers += 1;
                TensorSpec {
                    name: input.name.clone(),
                    buffer_id,
                    shape: input.shape.dims().to_vec(),
                    dtype: dtype_to_string(input.dtype),
                }
            })
            .collect();

        // Allocate buffers for constants
        let constants: Vec<ConstantData> = snapshot
            .constants
            .iter()
            .map(|constant| {
                let buffer_id = num_buffers;
                tensor_to_buffer.insert(constant.id.0, buffer_id);
                num_buffers += 1;
                ConstantData {
                    buffer_id,
                    shape: constant.shape.dims().to_vec(),
                    dtype: dtype_to_string(constant.dtype),
                    data: constant.data.clone(),
                }
            })
            .collect();

        // Generate dispatches for each node
        for node in &snapshot.nodes {
            let output_buffer = num_buffers;
            tensor_to_buffer.insert(node.output_id.0, output_buffer);
            num_buffers += 1;

            let input_buffers: Vec<usize> = node
                .input_ids
                .iter()
                .map(|id| tensor_to_buffer.get(&id.0).copied().unwrap_or(0))
                .collect();

            let kernel_name = op_to_kernel_name(&node.op, node.output_dtype);
            let grid_size = node.output_layout.size();

            // Build metadata using hodu_core::op_metadatas
            let metadata = build_metadata(&node.op, node.params.as_ref(), &node.input_layouts, &node.output_layout);

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
        let outputs: Vec<TensorSpec> = snapshot
            .targets
            .iter()
            .map(|target| {
                let buffer_id = tensor_to_buffer.get(&target.id.0).copied().unwrap_or(0);
                let (shape, dtype) = tensor_info
                    .get(&target.id.0)
                    .cloned()
                    .unwrap_or_else(|| (Vec::new(), DType::F32));
                TensorSpec {
                    name: target.name.clone(),
                    buffer_id,
                    shape,
                    dtype: dtype_to_string(dtype),
                }
            })
            .collect();

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
    #[allow(dead_code)]
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

fn build_metadata(
    op: &Op,
    params: Option<&OpParams>,
    input_layouts: &[hodu_core::types::Layout],
    output_layout: &hodu_core::types::Layout,
) -> Vec<usize> {
    use hodu_core::op_metadatas;
    use hodu_core::op_params;

    match op {
        // Binary ops
        Op::Binary(_) => {
            if input_layouts.len() >= 2 {
                op_metadatas::binary_metadata(&input_layouts[0], &input_layouts[1], output_layout)
            } else {
                op_metadatas::unary_metadata(output_layout, output_layout)
            }
        },
        Op::BinaryLogical(_) => {
            if input_layouts.len() >= 2 {
                op_metadatas::binary_logical_metadata(&input_layouts[0], &input_layouts[1], output_layout)
            } else {
                op_metadatas::unary_metadata(output_layout, output_layout)
            }
        },
        Op::Cmp(_) => {
            if input_layouts.len() >= 2 {
                op_metadatas::cmp_metadata(&input_layouts[0], &input_layouts[1], output_layout)
            } else {
                op_metadatas::unary_metadata(output_layout, output_layout)
            }
        },
        Op::CmpScalar(_) => {
            if !input_layouts.is_empty() {
                op_metadatas::cmp_scalar_metadata(&input_layouts[0], output_layout)
            } else {
                op_metadatas::unary_metadata(output_layout, output_layout)
            }
        },

        // Unary ops
        Op::Unary(_) | Op::UnaryLogical(_) | Op::UnaryScalar(_) => {
            if !input_layouts.is_empty() {
                op_metadatas::unary_metadata(&input_layouts[0], output_layout)
            } else {
                op_metadatas::unary_metadata(output_layout, output_layout)
            }
        },

        // Matrix ops
        Op::Matrix(_) => {
            if input_layouts.len() >= 2 {
                op_metadatas::matmul_metadata(&input_layouts[0], &input_layouts[1], output_layout)
                    .unwrap_or_else(|_| op_metadatas::unary_metadata(output_layout, output_layout))
            } else {
                op_metadatas::unary_metadata(output_layout, output_layout)
            }
        },

        // Reduce ops
        Op::Reduce(_) => {
            if let Some(OpParams::Reduce(op_params::ReduceParams { dims, keep_dim })) = params {
                let dims_usize: Vec<usize> = dims.iter().map(|s| s.to_usize()).collect();
                if !input_layouts.is_empty() {
                    op_metadatas::reduce_metadata(&input_layouts[0], &dims_usize, *keep_dim)
                } else {
                    op_metadatas::unary_metadata(output_layout, output_layout)
                }
            } else {
                op_metadatas::unary_metadata(output_layout, output_layout)
            }
        },

        // Concat ops
        Op::Concat(_) => {
            if let Some(OpParams::Concat(op_params::ConcatParams { dim })) = params {
                let dim_usize = dim.to_usize();
                let layout_refs: Vec<&hodu_core::types::Layout> = input_layouts.iter().collect();
                op_metadatas::concat_metadata(&layout_refs, dim_usize, output_layout.shape().dims())
            } else {
                op_metadatas::unary_metadata(output_layout, output_layout)
            }
        },

        // Split ops
        Op::Split(_) => {
            if let Some(OpParams::Split(op_params::SplitParams {
                dim,
                sizes,
                output_index,
            })) = params
            {
                let dim_usize = dim.to_usize();
                // Calculate start position for this output
                let start: usize = sizes
                    .iter()
                    .take(*output_index)
                    .map(|s| s.to_usize())
                    .collect::<Vec<_>>()
                    .iter()
                    .sum();
                let size = sizes.get(*output_index).map(|s| s.to_usize()).unwrap_or(0);
                if !input_layouts.is_empty() {
                    op_metadatas::split_metadata(&input_layouts[0], dim_usize, size, start, output_layout.size())
                } else {
                    op_metadatas::unary_metadata(output_layout, output_layout)
                }
            } else {
                op_metadatas::unary_metadata(output_layout, output_layout)
            }
        },

        // Indexing ops
        Op::Indexing(indexing_op) => {
            use hodu_core::ops::IndexingOp;
            match indexing_op {
                IndexingOp::IndexSelect => {
                    if let Some(OpParams::IndexSelect(op_params::IndexSelectParams { dim })) = params {
                        // num_indices can be inferred from indices tensor (input_layouts[1])
                        let num_indices = if input_layouts.len() >= 2 {
                            input_layouts[1].size()
                        } else {
                            0
                        };
                        if !input_layouts.is_empty() {
                            op_metadatas::index_select_metadata(
                                &input_layouts[0],
                                dim.to_usize(),
                                num_indices,
                                output_layout.size(),
                            )
                        } else {
                            op_metadatas::unary_metadata(output_layout, output_layout)
                        }
                    } else {
                        op_metadatas::unary_metadata(output_layout, output_layout)
                    }
                },
                IndexingOp::IndexPut => {
                    if let Some(OpParams::IndexPut(op_params::IndexPutParams { dim })) = params {
                        let num_indices = if input_layouts.len() >= 2 {
                            input_layouts[1].size()
                        } else {
                            0
                        };
                        if input_layouts.len() >= 2 {
                            op_metadatas::index_put_metadata(
                                &input_layouts[0],
                                &input_layouts[1],
                                dim.to_usize(),
                                num_indices,
                                output_layout.size(),
                            )
                        } else {
                            op_metadatas::unary_metadata(output_layout, output_layout)
                        }
                    } else {
                        op_metadatas::unary_metadata(output_layout, output_layout)
                    }
                },
                IndexingOp::Gather => {
                    if let Some(OpParams::Gather(op_params::GatherParams { dim })) = params {
                        if input_layouts.len() >= 2 {
                            op_metadatas::gather_metadata(
                                &input_layouts[0],
                                &input_layouts[1],
                                dim.to_usize(),
                                output_layout.size(),
                            )
                        } else {
                            op_metadatas::unary_metadata(output_layout, output_layout)
                        }
                    } else {
                        op_metadatas::unary_metadata(output_layout, output_layout)
                    }
                },
                IndexingOp::Scatter | IndexingOp::ScatterAdd | IndexingOp::ScatterMax | IndexingOp::ScatterMin => {
                    let dim = match params {
                        Some(OpParams::Scatter(p)) => p.dim.to_usize(),
                        Some(OpParams::ScatterAdd(p)) => p.dim.to_usize(),
                        Some(OpParams::ScatterMax(p)) => p.dim.to_usize(),
                        Some(OpParams::ScatterMin(p)) => p.dim.to_usize(),
                        _ => 0,
                    };
                    if input_layouts.len() >= 3 {
                        op_metadatas::scatter_metadata(&input_layouts[0], &input_layouts[1], &input_layouts[2], dim)
                    } else {
                        op_metadatas::unary_metadata(output_layout, output_layout)
                    }
                },
            }
        },

        // Conv ops
        Op::Conv(conv_op) => {
            use hodu_core::ops::ConvOp;
            match conv_op {
                ConvOp::Conv1d => {
                    if let Some(OpParams::Conv1d(p)) = params {
                        if input_layouts.len() >= 2 {
                            op_metadatas::conv1d_metadata(
                                &input_layouts[0],
                                &input_layouts[1],
                                p.stride,
                                p.padding,
                                p.dilation,
                                output_layout.shape().dims(),
                            )
                        } else {
                            op_metadatas::unary_metadata(output_layout, output_layout)
                        }
                    } else {
                        op_metadatas::unary_metadata(output_layout, output_layout)
                    }
                },
                ConvOp::Conv2d => {
                    if let Some(OpParams::Conv2d(p)) = params {
                        if input_layouts.len() >= 2 {
                            op_metadatas::conv2d_metadata(
                                &input_layouts[0],
                                &input_layouts[1],
                                &[p.stride, p.stride],
                                &[p.padding, p.padding],
                                &[p.dilation, p.dilation],
                                output_layout.shape().dims(),
                            )
                        } else {
                            op_metadatas::unary_metadata(output_layout, output_layout)
                        }
                    } else {
                        op_metadatas::unary_metadata(output_layout, output_layout)
                    }
                },
                ConvOp::Conv3d => {
                    if let Some(OpParams::Conv3d(p)) = params {
                        if input_layouts.len() >= 2 {
                            op_metadatas::conv3d_metadata(
                                &input_layouts[0],
                                &input_layouts[1],
                                &[p.stride, p.stride, p.stride],
                                &[p.padding, p.padding, p.padding],
                                &[p.dilation, p.dilation, p.dilation],
                                output_layout.shape().dims(),
                            )
                        } else {
                            op_metadatas::unary_metadata(output_layout, output_layout)
                        }
                    } else {
                        op_metadatas::unary_metadata(output_layout, output_layout)
                    }
                },
                // Transpose and gradient convolutions - use unary for now
                _ => op_metadatas::unary_metadata(output_layout, output_layout),
            }
        },

        // Windowing ops
        Op::Windowing(_) => {
            if let Some(OpParams::ReduceWindow(op_params::ReduceWindowParams {
                window_shape,
                strides,
                padding,
                ..
            })) = params
            {
                if !input_layouts.is_empty() {
                    // Flatten padding from [(lo, hi), ...] to [lo, hi, lo, hi, ...]
                    let flat_padding: Vec<usize> = padding.iter().flat_map(|(lo, hi)| vec![*lo, *hi]).collect();
                    op_metadatas::reduce_window_metadata(
                        &input_layouts[0],
                        window_shape,
                        strides,
                        &flat_padding,
                        output_layout.shape().dims(),
                    )
                } else {
                    op_metadatas::unary_metadata(output_layout, output_layout)
                }
            } else {
                op_metadatas::unary_metadata(output_layout, output_layout)
            }
        },

        // Shape ops, Cast ops, Memory ops - no kernels needed or use unary
        Op::Shape(_) | Op::ShapeScalars(_) | Op::Cast(_) | Op::Memory(_) | Op::Dummy => {
            op_metadatas::unary_metadata(output_layout, output_layout)
        },
    }
}
