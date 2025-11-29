//! Hodu Snapshot to ONNX export

use crate::onnx::{self, GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto};
use hodu_core::{
    error::{HoduError, HoduResult},
    ops::*,
    snapshot::{Snapshot, SnapshotTensorId},
    types::DType,
};
use std::collections::HashMap;

const IR_VERSION: i64 = 9;
const OPSET_VERSION: i64 = 19;

/// Convert Hodu Snapshot to ONNX ModelProto
pub fn export_model(snapshot: &Snapshot) -> HoduResult<ModelProto> {
    let graph = export_graph(snapshot)?;

    Ok(ModelProto {
        ir_version: IR_VERSION,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: OPSET_VERSION,
        }],
        producer_name: "hodu".to_string(),
        producer_version: env!("CARGO_PKG_VERSION").to_string(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: vec![],
    })
}

/// Convert Snapshot to ONNX GraphProto
fn export_graph(snapshot: &Snapshot) -> HoduResult<GraphProto> {
    let mut nodes = Vec::new();
    let mut initializers = Vec::new();

    // Build tensor ID to name mapping
    let mut name_map: HashMap<SnapshotTensorId, String> = HashMap::new();

    // Map input tensor IDs to their names
    for input in &snapshot.inputs {
        name_map.insert(input.id, input.name.clone());
    }

    // Map constant tensor IDs to their names
    for constant in &snapshot.constants {
        let name = constant
            .name
            .clone()
            .unwrap_or_else(|| format!("const_{}", constant.id.0));
        name_map.insert(constant.id, name);
    }

    // Map node output tensor IDs to names (use target name if it's a target, else generate)
    for node in &snapshot.nodes {
        if !name_map.contains_key(&node.output_id) {
            // Check if this is a target output
            let target_name = snapshot
                .targets
                .iter()
                .find(|t| t.id == node.output_id)
                .map(|t| t.name.clone());
            let name = target_name.unwrap_or_else(|| format!("t_{}", node.output_id.0));
            name_map.insert(node.output_id, name);
        }
    }

    // Convert inputs
    let inputs: Vec<ValueInfoProto> = snapshot
        .inputs
        .iter()
        .map(|input| make_value_info(&input.name, &input.shape.dims(), input.dtype))
        .collect();

    // Convert constants to initializers
    for constant in &snapshot.constants {
        let name = name_map.get(&constant.id).unwrap();
        initializers.push(make_tensor_proto(
            name,
            &constant.shape.dims(),
            constant.dtype,
            &constant.data,
        ));
    }

    // Convert nodes
    for node in &snapshot.nodes {
        if let Some(onnx_node) = convert_node(node, &name_map)? {
            nodes.push(onnx_node);
        }
    }

    // Convert outputs
    let outputs: Vec<ValueInfoProto> = snapshot
        .targets
        .iter()
        .map(|target| {
            // Find output info from nodes
            let node = snapshot.nodes.iter().find(|n| n.output_id == target.id);
            let (shape, dtype) = if let Some(n) = node {
                (n.output_layout.shape().dims().to_vec(), n.output_dtype)
            } else {
                (vec![], DType::F32)
            };
            make_value_info(&target.name, &shape, dtype)
        })
        .collect();

    Ok(GraphProto {
        name: snapshot.name.clone().unwrap_or_else(|| "hodu_graph".to_string()),
        node: nodes,
        initializer: initializers,
        input: inputs,
        output: outputs,
        value_info: vec![],
        doc_string: String::new(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
        metadata_props: vec![],
    })
}

/// Convert Hodu SnapshotNode to ONNX NodeProto
fn convert_node(
    node: &hodu_core::snapshot::SnapshotNode,
    name_map: &HashMap<SnapshotTensorId, String>,
) -> HoduResult<Option<NodeProto>> {
    let input_names: Vec<String> = node
        .input_ids
        .iter()
        .map(|id| name_map.get(id).cloned().unwrap_or_else(|| format!("t_{}", id.0)))
        .collect();
    let output_name = name_map
        .get(&node.output_id)
        .cloned()
        .unwrap_or_else(|| format!("t_{}", node.output_id.0));

    let (op_type, attributes) = match &node.op {
        Op::Binary(op) => {
            let op_type = match op {
                BinaryOp::Add => "Add",
                BinaryOp::Sub => "Sub",
                BinaryOp::Mul => "Mul",
                BinaryOp::Div => "Div",
                BinaryOp::Pow => "Pow",
                BinaryOp::Maximum => "Max",
                BinaryOp::Minimum => "Min",
            };
            (op_type, vec![])
        },

        Op::Unary(op) => {
            let op_type = match op {
                UnaryOp::Neg => "Neg",
                UnaryOp::Abs => "Abs",
                UnaryOp::Sign => "Sign",
                UnaryOp::Sqrt => "Sqrt",
                UnaryOp::Recip => "Reciprocal",
                UnaryOp::Relu => "Relu",
                UnaryOp::Sigmoid => "Sigmoid",
                UnaryOp::Tanh => "Tanh",
                UnaryOp::Exp => "Exp",
                UnaryOp::Ln => "Log",
                UnaryOp::Sin => "Sin",
                UnaryOp::Cos => "Cos",
                UnaryOp::Tan => "Tan",
                UnaryOp::Gelu => "Gelu",
                UnaryOp::Softplus => "Softplus",
                _ => {
                    return Err(HoduError::UnsupportedOperation(format!(
                        "Unary op {:?} not exportable to ONNX",
                        op
                    )))
                },
            };
            (op_type, vec![])
        },

        Op::Cmp(op) => {
            let op_type = match op {
                CmpOp::Eq => "Equal",
                CmpOp::Ne => return Err(HoduError::UnsupportedOperation("Ne not directly in ONNX".into())),
                CmpOp::Lt => "Less",
                CmpOp::Le => "LessOrEqual",
                CmpOp::Gt => "Greater",
                CmpOp::Ge => "GreaterOrEqual",
            };
            (op_type, vec![])
        },

        Op::BinaryLogical(op) => {
            let op_type = match op {
                BinaryLogicalOp::LogicalAnd => "And",
                BinaryLogicalOp::LogicalOr => "Or",
                BinaryLogicalOp::LogicalXor => "Xor",
            };
            (op_type, vec![])
        },

        Op::UnaryLogical(op) => {
            let op_type = match op {
                UnaryLogicalOp::LogicalNot => "Not",
            };
            (op_type, vec![])
        },

        Op::Matrix(op) => {
            let op_type = match op {
                MatrixOp::Matmul => "MatMul",
                MatrixOp::Dot => "MatMul",
            };
            (op_type, vec![])
        },

        Op::Reduce(op) => {
            let op_type = match op {
                ReduceOp::Sum => "ReduceSum",
                ReduceOp::Mean => "ReduceMean",
                ReduceOp::Max => "ReduceMax",
                ReduceOp::Min => "ReduceMin",
                ReduceOp::Prod => "ReduceProd",
                ReduceOp::ArgMax => "ArgMax",
                ReduceOp::ArgMin => "ArgMin",
                _ => {
                    return Err(HoduError::UnsupportedOperation(format!(
                        "Reduce op {:?} not exportable",
                        op
                    )))
                },
            };

            let mut attrs = vec![];
            if let Some(OpParams::Reduce(params)) = &node.params {
                let axes: Vec<i64> = params.dims.iter().map(|s| s.to_i32() as i64).collect();
                if !axes.is_empty() {
                    attrs.push(make_attr_ints("axes", axes));
                }
                attrs.push(make_attr_int("keepdims", if params.keep_dim { 1 } else { 0 }));
            }
            (op_type, attrs)
        },

        Op::Shape(op) => {
            let op_type = match op {
                ShapeOp::Reshape => "Reshape",
                ShapeOp::Flatten => "Flatten",
                ShapeOp::Squeeze => "Squeeze",
                ShapeOp::Unsqueeze => "Unsqueeze",
                ShapeOp::Transpose => "Transpose",
                ShapeOp::Permute => "Transpose",
                ShapeOp::Broadcast => {
                    return Err(HoduError::UnsupportedOperation("Broadcast not directly in ONNX".into()))
                },
            };
            (op_type, vec![])
        },

        Op::Concat(_) => {
            let mut attrs = vec![];
            if let Some(OpParams::Concat(params)) = &node.params {
                attrs.push(make_attr_int("axis", params.dim.to_i32() as i64));
            }
            ("Concat", attrs)
        },

        Op::Indexing(op) => {
            let op_type = match op {
                IndexingOp::Gather => "Gather",
                _ => {
                    return Err(HoduError::UnsupportedOperation(format!(
                        "Indexing op {:?} not exportable",
                        op
                    )))
                },
            };

            let mut attrs = vec![];
            if let Some(OpParams::Gather(params)) = &node.params {
                attrs.push(make_attr_int("axis", params.dim.to_i32() as i64));
            }
            (op_type, attrs)
        },

        Op::Cast(_) => {
            let mut attrs = vec![];
            if let Some(OpParams::ToDType(params)) = &node.params {
                attrs.push(make_attr_int("to", hodu_dtype_to_onnx(params.dtype) as i64));
            }
            ("Cast", attrs)
        },

        Op::UnaryScalar(op) => match op {
            UnaryScalarOp::LeakyRelu => {
                let mut attrs = vec![];
                if let Some(OpParams::UnaryScalar(params)) = &node.params {
                    attrs.push(make_attr_float("alpha", params.scalar.to_f32()));
                }
                ("LeakyRelu", attrs)
            },
            _ => {
                return Err(HoduError::UnsupportedOperation(format!(
                    "UnaryScalar op {:?} not exportable",
                    op
                )))
            },
        },

        Op::Dummy => return Ok(None),

        _ => {
            return Err(HoduError::UnsupportedOperation(format!(
                "Op {:?} not exportable to ONNX",
                node.op
            )))
        },
    };

    Ok(Some(NodeProto {
        input: input_names,
        output: vec![output_name],
        name: String::new(),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute: attributes,
        doc_string: String::new(),
        overload: String::new(),
        metadata_props: vec![],
        device_configurations: vec![],
    }))
}

fn make_value_info(name: &str, shape: &[usize], dtype: DType) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(onnx::TypeProto {
            denotation: String::new(),
            value: Some(onnx::type_proto::Value::TensorType(onnx::type_proto::Tensor {
                elem_type: hodu_dtype_to_onnx(dtype),
                shape: Some(onnx::TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|&d| onnx::tensor_shape_proto::Dimension {
                            denotation: String::new(),
                            value: Some(onnx::tensor_shape_proto::dimension::Value::DimValue(d as i64)),
                        })
                        .collect(),
                }),
            })),
        }),
        doc_string: String::new(),
        metadata_props: vec![],
    }
}

fn make_tensor_proto(name: &str, shape: &[usize], dtype: DType, data: &[u8]) -> TensorProto {
    TensorProto {
        dims: shape.iter().map(|&d| d as i64).collect(),
        data_type: hodu_dtype_to_onnx(dtype),
        segment: None,
        float_data: vec![],
        int32_data: vec![],
        string_data: vec![],
        int64_data: vec![],
        name: name.to_string(),
        doc_string: String::new(),
        raw_data: data.to_vec(),
        external_data: vec![],
        data_location: 0,
        double_data: vec![],
        uint64_data: vec![],
        metadata_props: vec![],
    }
}

fn make_attr_int(name: &str, value: i64) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        ref_attr_name: String::new(),
        doc_string: String::new(),
        r#type: onnx::attribute_proto::AttributeType::Int as i32,
        f: 0.0,
        i: value,
        s: vec![],
        t: None,
        g: None,
        sparse_tensor: None,
        tp: None,
        floats: vec![],
        ints: vec![],
        strings: vec![],
        tensors: vec![],
        graphs: vec![],
        sparse_tensors: vec![],
        type_protos: vec![],
    }
}

fn make_attr_ints(name: &str, values: Vec<i64>) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        ref_attr_name: String::new(),
        doc_string: String::new(),
        r#type: onnx::attribute_proto::AttributeType::Ints as i32,
        f: 0.0,
        i: 0,
        s: vec![],
        t: None,
        g: None,
        sparse_tensor: None,
        tp: None,
        floats: vec![],
        ints: values,
        strings: vec![],
        tensors: vec![],
        graphs: vec![],
        sparse_tensors: vec![],
        type_protos: vec![],
    }
}

fn make_attr_float(name: &str, value: f32) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        ref_attr_name: String::new(),
        doc_string: String::new(),
        r#type: onnx::attribute_proto::AttributeType::Float as i32,
        f: value,
        i: 0,
        s: vec![],
        t: None,
        g: None,
        sparse_tensor: None,
        tp: None,
        floats: vec![],
        ints: vec![],
        strings: vec![],
        tensors: vec![],
        graphs: vec![],
        sparse_tensors: vec![],
        type_protos: vec![],
    }
}

fn hodu_dtype_to_onnx(dtype: DType) -> i32 {
    match dtype {
        DType::F32 => 1,
        DType::F16 => 10,
        DType::BF16 => 16,
        DType::F8E4M3 => 17,
        DType::BOOL => 9,
        DType::I8 => 3,
        DType::I32 => 6,
        DType::U8 => 2,
        DType::U32 => 12,
        #[allow(unreachable_patterns)]
        _ => 1, // default to float
    }
}
