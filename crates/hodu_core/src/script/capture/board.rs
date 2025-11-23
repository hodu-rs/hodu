use crate::{
    compat::*,
    error::HoduResult,
    ops::{Op, OpParams},
    script::{Script, Snapshot, SnapshotInput, SnapshotNode, SnapshotTarget, SnapshotTensorId},
    tensor::{Tensor, TensorId},
    types::{DType, Layout, Shape},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CaptureBoardId(usize);

impl CaptureBoardId {
    pub(super) fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// A captured input tensor
#[derive(Debug, Clone)]
pub struct CapturedInput {
    pub name: String,
    pub tensor_id: TensorId,
    pub shape: Shape,
    pub dtype: DType,
}

/// A captured output tensor
#[derive(Debug, Clone)]
pub struct CapturedTarget {
    pub name: String,
    pub tensor_id: TensorId,
}

/// A captured operation with its metadata
#[derive(Debug, Clone)]
pub struct CapturedOp {
    pub op: Op,
    pub params: Option<OpParams>,
    pub input_ids: Vec<TensorId>,
    pub output_id: TensorId,
    pub input_layouts: Vec<Layout>,
    pub output_layout: Layout,
}

/// CaptureBoard captures tensor operations to build a Script
pub struct CaptureBoard {
    pub(super) id: CaptureBoardId,
    name: Option<String>,
    inputs: Vec<CapturedInput>,
    targets: Vec<CapturedTarget>,
    ops: Vec<CapturedOp>,
}

impl CaptureBoard {
    pub fn new() -> Self {
        Self {
            id: CaptureBoardId::new(),
            name: None,
            inputs: Vec::new(),
            targets: Vec::new(),
            ops: Vec::new(),
        }
    }

    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            id: CaptureBoardId::new(),
            name: Some(name.into()),
            inputs: Vec::new(),
            targets: Vec::new(),
            ops: Vec::new(),
        }
    }

    pub fn id(&self) -> CaptureBoardId {
        self.id
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn inputs(&self) -> &[CapturedInput] {
        &self.inputs
    }

    pub fn targets(&self) -> &[CapturedTarget] {
        &self.targets
    }

    pub fn ops(&self) -> &[CapturedOp] {
        &self.ops
    }

    pub fn into_parts(self) -> (Vec<CapturedInput>, Vec<CapturedTarget>, Vec<CapturedOp>) {
        (self.inputs, self.targets, self.ops)
    }

    pub(crate) fn add_input(&mut self, name: &str, tensor: Tensor) -> HoduResult<()> {
        let input = CapturedInput {
            name: name.to_string(),
            tensor_id: tensor.id(),
            shape: tensor.shape(),
            dtype: tensor.dtype(),
        };
        self.inputs.push(input);
        Ok(())
    }

    pub fn add_target(&mut self, name: &str, tensor: Tensor) {
        let target = CapturedTarget {
            name: name.to_string(),
            tensor_id: tensor.id(),
        };
        self.targets.push(target);
    }

    pub(super) fn add_op(&mut self, op: CapturedOp) {
        self.ops.push(op);
    }

    /// Convert captured operations into a Script
    pub fn capture(self) -> Script {
        // Dead code elimination: mark reachable nodes from targets
        let mut reachable = HashSet::new();
        let mut to_visit = Vec::new();

        // Start from target tensor IDs
        for target in &self.targets {
            to_visit.push(target.tensor_id);
        }

        // Backward traversal to find all reachable nodes
        while let Some(tensor_id) = to_visit.pop() {
            if !reachable.insert(tensor_id) {
                continue; // Already visited
            }

            // Find the op that produces this tensor
            for op in &self.ops {
                if op.output_id == tensor_id {
                    // Mark all input tensors for visiting
                    for input_id in &op.input_ids {
                        to_visit.push(*input_id);
                    }
                    break;
                }
            }
        }

        // Filter ops: keep only reachable ones
        let filtered_ops: Vec<_> = self
            .ops
            .into_iter()
            .filter(|op| reachable.contains(&op.output_id))
            .collect();

        // Find the minimum TensorId to normalize all IDs to start from 0
        let mut min_id = usize::MAX;

        for input in &self.inputs {
            min_id = min_id.min(input.tensor_id.as_usize());
        }

        for target in &self.targets {
            min_id = min_id.min(target.tensor_id.as_usize());
        }

        for op in &filtered_ops {
            for input_id in &op.input_ids {
                min_id = min_id.min(input_id.as_usize());
            }
            min_id = min_id.min(op.output_id.as_usize());
        }

        // If no tensors were captured, use 0 as the offset
        let offset = if min_id == usize::MAX { 0 } else { min_id };

        // Convert captured inputs to snapshot inputs with normalized IDs
        let snapshot_inputs = self
            .inputs
            .into_iter()
            .map(|input| SnapshotInput {
                name: input.name,
                id: SnapshotTensorId(input.tensor_id.as_usize() - offset),
                shape: input.shape,
                dtype: input.dtype,
            })
            .collect();

        // Convert captured targets to snapshot targets with normalized IDs
        let snapshot_targets = self
            .targets
            .into_iter()
            .map(|target| SnapshotTarget {
                name: target.name,
                id: SnapshotTensorId(target.tensor_id.as_usize() - offset),
            })
            .collect();

        // Convert filtered ops to snapshot nodes with normalized IDs
        let snapshot_nodes = filtered_ops
            .into_iter()
            .map(|op| {
                // Get output dtype from the tensor registry
                let output_dtype = crate::tensor::get_dtype(op.output_id).expect("Builder tensor must have dtype");

                SnapshotNode {
                    op: op.op,
                    params: op.params,
                    input_ids: op
                        .input_ids
                        .into_iter()
                        .map(|id| SnapshotTensorId(id.as_usize() - offset))
                        .collect(),
                    output_id: SnapshotTensorId(op.output_id.as_usize() - offset),
                    input_layouts: op.input_layouts,
                    output_layout: op.output_layout,
                    output_dtype,
                }
            })
            .collect();

        let snapshot = Snapshot {
            name: self.name,
            inputs: snapshot_inputs,
            targets: snapshot_targets,
            nodes: snapshot_nodes,
        };

        Script::new(snapshot)
    }
}

impl Default for CaptureBoard {
    fn default() -> Self {
        Self::new()
    }
}
