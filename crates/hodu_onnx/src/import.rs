//! ONNX to Hodu Snapshot import

use crate::onnx::{GraphProto, ModelProto, NodeProto, TensorProto};
use hodu_core::{
    error::{HoduError, HoduResult},
    ops::*,
    scalar::Scalar,
    snapshot::{Snapshot, SnapshotConstant, SnapshotInput, SnapshotNode, SnapshotTarget, SnapshotTensorId},
    types::{DType, Layout, Shape},
};
use std::collections::HashMap;

/// Import ONNX ModelProto to Hodu Snapshot
pub fn import_model(model: &ModelProto) -> HoduResult<Snapshot> {
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| HoduError::InvalidArgument("ONNX model has no graph".into()))?;

    convert_graph(graph, model.producer_name.as_str())
}

/// Convert ONNX GraphProto to Hodu Snapshot
fn convert_graph(graph: &GraphProto, producer: &str) -> HoduResult<Snapshot> {
    let mut snapshot = Snapshot::new();
    snapshot.name = if graph.name.is_empty() {
        Some(format!("onnx_{}", producer))
    } else {
        Some(graph.name.clone())
    };

    // Track tensor name to ID mapping
    let mut tensor_map: HashMap<String, SnapshotTensorId> = HashMap::new();
    let mut next_id = 0usize;

    let mut alloc_id = || {
        let id = SnapshotTensorId(next_id);
        next_id += 1;
        id
    };

    // Collect initializer names (these are weights, not inputs)
    let initializer_names: std::collections::HashSet<_> = graph.initializer.iter().map(|t| t.name.as_str()).collect();

    // Process inputs (excluding initializers)
    for input in &graph.input {
        if initializer_names.contains(input.name.as_str()) {
            continue;
        }

        let id = alloc_id();
        tensor_map.insert(input.name.clone(), id);

        let (shape, dtype) = parse_value_info(input)?;
        snapshot.inputs.push(SnapshotInput {
            name: input.name.clone(),
            id,
            shape,
            dtype,
        });
    }

    // Process initializers as constants
    for init in &graph.initializer {
        let id = alloc_id();
        tensor_map.insert(init.name.clone(), id);

        let (shape, dtype, data) = parse_tensor(init)?;
        snapshot.constants.push(SnapshotConstant {
            id,
            name: Some(init.name.clone()),
            shape,
            dtype,
            data,
        });
    }

    // Process nodes
    for node in &graph.node {
        convert_node(node, &mut snapshot, &mut tensor_map, &mut alloc_id)?;
    }

    // Process outputs
    for output in &graph.output {
        let id = tensor_map
            .get(&output.name)
            .copied()
            .ok_or_else(|| HoduError::InvalidArgument(format!("Output '{}' not found", output.name)))?;

        snapshot.targets.push(SnapshotTarget {
            name: output.name.clone(),
            id,
        });
    }

    Ok(snapshot)
}

/// Convert ONNX node to Hodu SnapshotNode(s)
fn convert_node<F>(
    node: &NodeProto,
    snapshot: &mut Snapshot,
    tensor_map: &mut HashMap<String, SnapshotTensorId>,
    alloc_id: &mut F,
) -> HoduResult<()>
where
    F: FnMut() -> SnapshotTensorId,
{
    let op_type = node.op_type.as_str();

    // Get input IDs
    let input_ids: Vec<SnapshotTensorId> = node
        .input
        .iter()
        .filter(|name| !name.is_empty())
        .map(|name| {
            tensor_map
                .get(name)
                .copied()
                .ok_or_else(|| HoduError::InvalidArgument(format!("Input '{}' not found for op {}", name, op_type)))
        })
        .collect::<HoduResult<Vec<_>>>()?;

    // Convert based on op type
    let (op, params, output_dtype) = match op_type {
        // Binary ops
        "Add" => (Op::Binary(BinaryOp::Add), None, None),
        "Sub" => (Op::Binary(BinaryOp::Sub), None, None),
        "Mul" => (Op::Binary(BinaryOp::Mul), None, None),
        "Div" => (Op::Binary(BinaryOp::Div), None, None),
        "Pow" => (Op::Binary(BinaryOp::Pow), None, None),
        "Max" => (Op::Binary(BinaryOp::Maximum), None, None),
        "Min" => (Op::Binary(BinaryOp::Minimum), None, None),

        // Unary ops
        "Neg" => (Op::Unary(UnaryOp::Neg), None, None),
        "Abs" => (Op::Unary(UnaryOp::Abs), None, None),
        "Sign" => (Op::Unary(UnaryOp::Sign), None, None),
        "Sqrt" => (Op::Unary(UnaryOp::Sqrt), None, None),
        "Reciprocal" => (Op::Unary(UnaryOp::Recip), None, None),
        "Relu" => (Op::Unary(UnaryOp::Relu), None, None),
        "Sigmoid" => (Op::Unary(UnaryOp::Sigmoid), None, None),
        "Tanh" => (Op::Unary(UnaryOp::Tanh), None, None),
        "Exp" => (Op::Unary(UnaryOp::Exp), None, None),
        "Log" => (Op::Unary(UnaryOp::Ln), None, None),
        "Sin" => (Op::Unary(UnaryOp::Sin), None, None),
        "Cos" => (Op::Unary(UnaryOp::Cos), None, None),
        "Tan" => (Op::Unary(UnaryOp::Tan), None, None),
        "Gelu" => (Op::Unary(UnaryOp::Gelu), None, None),
        "Softplus" => (Op::Unary(UnaryOp::Softplus), None, None),

        // Comparison ops
        "Equal" => (Op::Cmp(CmpOp::Eq), None, Some(DType::BOOL)),
        "Less" => (Op::Cmp(CmpOp::Lt), None, Some(DType::BOOL)),
        "LessOrEqual" => (Op::Cmp(CmpOp::Le), None, Some(DType::BOOL)),
        "Greater" => (Op::Cmp(CmpOp::Gt), None, Some(DType::BOOL)),
        "GreaterOrEqual" => (Op::Cmp(CmpOp::Ge), None, Some(DType::BOOL)),

        // Logical ops
        "And" => (Op::BinaryLogical(BinaryLogicalOp::LogicalAnd), None, Some(DType::BOOL)),
        "Or" => (Op::BinaryLogical(BinaryLogicalOp::LogicalOr), None, Some(DType::BOOL)),
        "Xor" => (Op::BinaryLogical(BinaryLogicalOp::LogicalXor), None, Some(DType::BOOL)),
        "Not" => (Op::UnaryLogical(UnaryLogicalOp::LogicalNot), None, Some(DType::BOOL)),

        // Matrix ops
        "MatMul" => (Op::Matrix(MatrixOp::Matmul), None, None),
        "Gemm" => (Op::Matrix(MatrixOp::Matmul), None, None),

        // Reduce ops
        "ReduceSum" => {
            let axes = get_attr_ints(node, "axes").unwrap_or_default();
            let keepdims = get_attr_int(node, "keepdims").unwrap_or(1) == 1;
            (
                Op::Reduce(ReduceOp::Sum),
                Some(OpParams::Reduce(ReduceParams {
                    dims: axes.iter().map(|&a| Scalar::I32(a as i32)).collect(),
                    keep_dim: keepdims,
                })),
                None,
            )
        },
        "ReduceMean" => {
            let axes = get_attr_ints(node, "axes").unwrap_or_default();
            let keepdims = get_attr_int(node, "keepdims").unwrap_or(1) == 1;
            (
                Op::Reduce(ReduceOp::Mean),
                Some(OpParams::Reduce(ReduceParams {
                    dims: axes.iter().map(|&a| Scalar::I32(a as i32)).collect(),
                    keep_dim: keepdims,
                })),
                None,
            )
        },
        "ReduceMax" => {
            let axes = get_attr_ints(node, "axes").unwrap_or_default();
            let keepdims = get_attr_int(node, "keepdims").unwrap_or(1) == 1;
            (
                Op::Reduce(ReduceOp::Max),
                Some(OpParams::Reduce(ReduceParams {
                    dims: axes.iter().map(|&a| Scalar::I32(a as i32)).collect(),
                    keep_dim: keepdims,
                })),
                None,
            )
        },
        "ReduceMin" => {
            let axes = get_attr_ints(node, "axes").unwrap_or_default();
            let keepdims = get_attr_int(node, "keepdims").unwrap_or(1) == 1;
            (
                Op::Reduce(ReduceOp::Min),
                Some(OpParams::Reduce(ReduceParams {
                    dims: axes.iter().map(|&a| Scalar::I32(a as i32)).collect(),
                    keep_dim: keepdims,
                })),
                None,
            )
        },
        "ReduceProd" => {
            let axes = get_attr_ints(node, "axes").unwrap_or_default();
            let keepdims = get_attr_int(node, "keepdims").unwrap_or(1) == 1;
            (
                Op::Reduce(ReduceOp::Prod),
                Some(OpParams::Reduce(ReduceParams {
                    dims: axes.iter().map(|&a| Scalar::I32(a as i32)).collect(),
                    keep_dim: keepdims,
                })),
                None,
            )
        },
        "ArgMax" => {
            let axis = get_attr_int(node, "axis").unwrap_or(0);
            let keepdims = get_attr_int(node, "keepdims").unwrap_or(1) == 1;
            (
                Op::Reduce(ReduceOp::ArgMax),
                Some(OpParams::Reduce(ReduceParams {
                    dims: vec![Scalar::I32(axis as i32)],
                    keep_dim: keepdims,
                })),
                Some(DType::I32),
            )
        },
        "ArgMin" => {
            let axis = get_attr_int(node, "axis").unwrap_or(0);
            let keepdims = get_attr_int(node, "keepdims").unwrap_or(1) == 1;
            (
                Op::Reduce(ReduceOp::ArgMin),
                Some(OpParams::Reduce(ReduceParams {
                    dims: vec![Scalar::I32(axis as i32)],
                    keep_dim: keepdims,
                })),
                Some(DType::I32),
            )
        },

        // Shape ops
        "Reshape" => (
            Op::Shape(ShapeOp::Reshape),
            Some(OpParams::Reshape(ReshapeParams)),
            None,
        ),
        "Flatten" => (
            Op::Shape(ShapeOp::Flatten),
            Some(OpParams::Flatten(FlattenParams)),
            None,
        ),
        "Squeeze" => (
            Op::Shape(ShapeOp::Squeeze),
            Some(OpParams::Squeeze(SqueezeParams)),
            None,
        ),
        "Unsqueeze" => (
            Op::Shape(ShapeOp::Unsqueeze),
            Some(OpParams::Unsqueeze(UnsqueezeParams)),
            None,
        ),
        "Transpose" => (
            Op::Shape(ShapeOp::Permute),
            Some(OpParams::Permute(PermuteParams)),
            None,
        ),

        // Concat
        "Concat" => {
            let axis = get_attr_int(node, "axis").unwrap_or(0);
            (
                Op::Concat(ConcatOp::Concat),
                Some(OpParams::Concat(ConcatParams {
                    dim: Scalar::I32(axis as i32),
                })),
                None,
            )
        },

        // Gather
        "Gather" => {
            let axis = get_attr_int(node, "axis").unwrap_or(0);
            (
                Op::Indexing(IndexingOp::Gather),
                Some(OpParams::Gather(GatherParams {
                    dim: Scalar::I32(axis as i32),
                })),
                None,
            )
        },

        // Cast
        "Cast" => {
            let to = get_attr_int(node, "to").unwrap_or(1);
            let target_dtype = onnx_dtype_to_hodu(to as i32)?;
            (
                Op::Cast(CastOp::ToDType),
                Some(OpParams::ToDType(ToDTypeParams { dtype: target_dtype })),
                Some(target_dtype),
            )
        },

        // LeakyRelu
        "LeakyRelu" => {
            let alpha = get_attr_float(node, "alpha").unwrap_or(0.01);
            (
                Op::UnaryScalar(UnaryScalarOp::LeakyRelu),
                Some(OpParams::UnaryScalar(UnaryScalarParams {
                    scalar: Scalar::F32(alpha),
                })),
                None,
            )
        },

        // Identity / Dropout (no-op in inference)
        "Identity" | "Dropout" => (Op::Dummy, None, None),

        // TODO: Conv, Pooling, BatchNorm, Softmax, etc.
        _ => {
            return Err(HoduError::UnsupportedOperation(format!(
                "ONNX op '{}' not supported",
                op_type
            )));
        },
    };

    // Create output tensor
    for (i, output_name) in node.output.iter().enumerate() {
        if output_name.is_empty() {
            continue;
        }

        let output_id = alloc_id();
        tensor_map.insert(output_name.clone(), output_id);

        if i == 0 {
            snapshot.nodes.push(SnapshotNode {
                op: op.clone(),
                params: params.clone(),
                input_ids: input_ids.clone(),
                output_id,
                input_layouts: vec![Layout::from_shape(&Shape::scalar()); input_ids.len()],
                output_layout: Layout::from_shape(&Shape::scalar()),
                output_dtype: output_dtype.unwrap_or(DType::F32),
            });
        }
    }

    Ok(())
}

/// Parse ONNX ValueInfoProto to shape and dtype
fn parse_value_info(info: &crate::onnx::ValueInfoProto) -> HoduResult<(Shape, DType)> {
    let tensor_type = info
        .r#type
        .as_ref()
        .and_then(|t| t.value.as_ref())
        .and_then(|v| match v {
            crate::onnx::type_proto::Value::TensorType(t) => Some(t),
            _ => None,
        })
        .ok_or_else(|| HoduError::InvalidArgument(format!("Input '{}' has no tensor type", info.name)))?;

    let dtype = onnx_dtype_to_hodu(tensor_type.elem_type)?;

    let shape = tensor_type
        .shape
        .as_ref()
        .map(|s| {
            s.dim
                .iter()
                .map(|d| match &d.value {
                    Some(crate::onnx::tensor_shape_proto::dimension::Value::DimValue(v)) => *v as usize,
                    Some(crate::onnx::tensor_shape_proto::dimension::Value::DimParam(_)) => 0,
                    None => 0,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok((Shape::new(&shape), dtype))
}

/// Parse ONNX TensorProto to shape, dtype, and raw data
fn parse_tensor(tensor: &TensorProto) -> HoduResult<(Shape, DType, Vec<u8>)> {
    let dtype = onnx_dtype_to_hodu(tensor.data_type)?;
    let shape: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();

    let data = if !tensor.raw_data.is_empty() {
        tensor.raw_data.clone()
    } else {
        match dtype {
            DType::F32 => tensor.float_data.iter().flat_map(|f| f.to_le_bytes()).collect(),
            DType::I32 => tensor.int32_data.iter().flat_map(|i| i.to_le_bytes()).collect(),
            _ => {
                return Err(HoduError::UnsupportedOperation(format!(
                    "Tensor data extraction for {:?} not implemented",
                    dtype
                )));
            },
        }
    };

    Ok((Shape::new(&shape), dtype, data))
}

/// Convert ONNX DataType to Hodu DType
fn onnx_dtype_to_hodu(dt: i32) -> HoduResult<DType> {
    const FLOAT: i32 = 1;
    const UINT8: i32 = 2;
    const INT8: i32 = 3;
    const INT32: i32 = 6;
    const BOOL: i32 = 9;
    const FLOAT16: i32 = 10;
    const BFLOAT16: i32 = 16;
    const FLOAT8E4M3FN: i32 = 17;
    const UINT32: i32 = 12;

    match dt {
        FLOAT => Ok(DType::F32),
        FLOAT16 => Ok(DType::F16),
        BFLOAT16 => Ok(DType::BF16),
        FLOAT8E4M3FN => Ok(DType::F8E4M3),
        BOOL => Ok(DType::BOOL),
        INT8 => Ok(DType::I8),
        INT32 => Ok(DType::I32),
        UINT8 => Ok(DType::U8),
        UINT32 => Ok(DType::U32),
        _ => Err(HoduError::UnsupportedOperation(format!(
            "ONNX DataType {} not supported",
            dt
        ))),
    }
}

fn get_attr_int(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute.iter().find(|a| a.name == name).and_then(|a| {
        if a.r#type == crate::onnx::attribute_proto::AttributeType::Int as i32 {
            Some(a.i)
        } else {
            None
        }
    })
}

fn get_attr_ints(node: &NodeProto, name: &str) -> Option<Vec<i64>> {
    node.attribute.iter().find(|a| a.name == name).and_then(|a| {
        if a.r#type == crate::onnx::attribute_proto::AttributeType::Ints as i32 {
            Some(a.ints.clone())
        } else {
            None
        }
    })
}

fn get_attr_float(node: &NodeProto, name: &str) -> Option<f32> {
    node.attribute.iter().find(|a| a.name == name).and_then(|a| {
        if a.r#type == crate::onnx::attribute_proto::AttributeType::Float as i32 {
            Some(a.f)
        } else {
            None
        }
    })
}
