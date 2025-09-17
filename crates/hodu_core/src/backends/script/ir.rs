use crate::{
    backends::op::Op,
    compat::*,
    tensor::TensorId,
    types::{dtype::DType, layout::Layout},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct NodeId(pub usize);

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct ScriptIR {
    pub version: String,
    pub metadata: ScriptMetadata,
    pub graph: ComputationGraph,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct ScriptMetadata {
    pub name: String,
    pub description: Option<String>,
    pub created_at: String,
    pub hodu_version: String,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct ComputationGraph {
    pub metadata: GraphMetadata,
    pub topology: GraphTopology,
    pub execution_plan: ExecutionPlan,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct GraphMetadata {
    pub inputs: Vec<InputNode>,
    pub outputs: Vec<OutputNode>,
    pub constants: HashMap<TensorId, ConstantNode>,
    pub tensor_info: HashMap<TensorId, TensorInfo>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct GraphTopology {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct ExecutionPlan {
    pub execution_order: Vec<NodeId>,
    pub optimization_hints: HashMap<String, String>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct GraphEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub tensor_id: TensorId,
    pub edge_type: EdgeType,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum EdgeType {
    Data,
    Control,
    Memory,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct GraphNode {
    pub id: NodeId,
    pub node_type: NodeType,
    pub operation: Op,
    pub attributes: HashMap<String, AttributeValue>,
    pub input_tensors: Vec<TensorId>,
    pub output_tensors: Vec<TensorId>,
    pub input_layouts: Vec<Layout>,
    pub output_layouts: Vec<Layout>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum NodeType {
    Compute,
    Control,
    Memory,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct InputNode {
    pub name: String,
    pub tensor_id: TensorId,
    pub optional: bool,
    pub default_value: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct OutputNode {
    pub name: String,
    pub tensor_id: TensorId,
    pub is_intermediate: bool,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct ConstantNode {
    pub tensor_id: TensorId,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<u8>,
    pub compression: Option<CompressionType>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub struct TensorInfo {
    pub id: TensorId,
    pub shape: Option<Vec<Option<usize>>>,
    pub dtype: Option<DType>,
    pub layout: Option<String>,
    pub memory_layout: Option<MemoryLayout>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum AttributeValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    IntArray(Vec<i64>),
    FloatArray(Vec<f64>),
    StringArray(Vec<String>),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum CompressionType {
    None,
    Gzip,
    Zstd,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", derive(bincode::Encode, bincode::Decode))]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
    Blocked(Vec<usize>),
    Custom(String),
}

impl ScriptIR {
    pub fn new(name: String) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            metadata: ScriptMetadata {
                name,
                description: None,
                #[cfg(feature = "std")]
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs().to_string())
                    .unwrap_or_else(|_| "0".to_string()),
                #[cfg(not(feature = "std"))]
                created_at: "0".to_string(),
                hodu_version: env!("CARGO_PKG_VERSION").to_string(),
            },
            graph: ComputationGraph::default(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.graph.metadata.inputs.is_empty() {
            return Err("Graph must have at least one input".to_string());
        }

        if self.graph.metadata.outputs.is_empty() {
            return Err("Graph must have at least one output".to_string());
        }

        let mut tensor_ids = HashSet::new();

        for input in &self.graph.metadata.inputs {
            if !tensor_ids.insert(input.tensor_id) {
                return Err(format!("Duplicate tensor ID in inputs: {:?}", input.tensor_id));
            }
        }

        for output in &self.graph.metadata.outputs {
            if !tensor_ids.insert(output.tensor_id) {
                return Err(format!("Duplicate tensor ID in outputs: {:?}", output.tensor_id));
            }
        }

        for tensor_id in self.graph.metadata.constants.keys() {
            if !tensor_ids.insert(*tensor_id) {
                return Err(format!("Duplicate tensor ID in constants: {tensor_id:?}"));
            }
        }

        for node in &self.graph.topology.nodes {
            for tensor_id in &node.input_tensors {
                if !tensor_ids.contains(tensor_id) && !self.graph.metadata.tensor_info.contains_key(tensor_id) {
                    return Err(format!(
                        "Node {:?} references unknown input tensor {:?}",
                        node.id, tensor_id
                    ));
                }
            }

            for tensor_id in &node.output_tensors {
                if !tensor_ids.contains(tensor_id) && !self.graph.metadata.tensor_info.contains_key(tensor_id) {
                    return Err(format!(
                        "Node {:?} references unknown output tensor {:?}",
                        node.id, tensor_id
                    ));
                }
            }
        }

        let node_ids: HashSet<_> = self.graph.topology.nodes.iter().map(|n| n.id).collect();
        for edge in &self.graph.topology.edges {
            if !node_ids.contains(&edge.from) {
                return Err(format!("Edge references unknown from node: {:?}", edge.from));
            }
            if !node_ids.contains(&edge.to) {
                return Err(format!("Edge references unknown to node: {:?}", edge.to));
            }
        }

        for node_id in &self.graph.execution_plan.execution_order {
            if !node_ids.contains(node_id) {
                return Err(format!("Execution plan references unknown node: {node_id:?}"));
            }
        }

        Ok(())
    }
}

impl ComputationGraph {
    pub fn new() -> Self {
        Self {
            metadata: GraphMetadata {
                inputs: Vec::new(),
                outputs: Vec::new(),
                constants: HashMap::new(),
                tensor_info: HashMap::new(),
            },
            topology: GraphTopology {
                nodes: Vec::new(),
                edges: Vec::new(),
            },
            execution_plan: ExecutionPlan {
                execution_order: Vec::new(),
                optimization_hints: HashMap::new(),
            },
        }
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

