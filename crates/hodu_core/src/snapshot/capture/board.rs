use crate::{
    error::HoduResult,
    ops::{Op, OpParams},
    snapshot::{Snapshot, SnapshotInput, SnapshotNode, SnapshotTarget, SnapshotTensorId},
    tensor::{Tensor, TensorId},
    types::{DType, Layout, Shape},
};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

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

/// Internal board data structure stored in global storage
pub(super) struct CaptureBoard_ {
    pub(super) id: CaptureBoardId,
    pub(super) name: Option<String>,
    pub(super) inputs: Vec<CapturedInput>,
    pub(super) targets: Vec<CapturedTarget>,
    pub(super) ops: Vec<CapturedOp>,
}

impl CaptureBoard_ {
    pub(super) fn add_input(&mut self, name: &str, tensor: Tensor) -> HoduResult<()> {
        let input = CapturedInput {
            name: name.to_string(),
            tensor_id: tensor.id(),
            shape: tensor.shape(),
            dtype: tensor.dtype(),
        };
        self.inputs.push(input);
        Ok(())
    }

    pub(super) fn add_target(&mut self, name: &str, tensor: Tensor) {
        let target = CapturedTarget {
            name: name.to_string(),
            tensor_id: tensor.id(),
        };
        self.targets.push(target);
    }

    pub(super) fn add_op(&mut self, op: CapturedOp) {
        self.ops.push(op);
    }
}

/// CaptureBoard is a handle that references a board in global storage
pub struct CaptureBoard(CaptureBoardId);

impl CaptureBoard {
    pub fn new() -> Self {
        let id = CaptureBoardId::new();
        let board = CaptureBoard_ {
            id,
            name: None,
            inputs: Vec::new(),
            targets: Vec::new(),
            ops: Vec::new(),
        };
        super::storage::register_board(board);
        Self(id)
    }

    pub fn with_name(name: impl Into<String>) -> Self {
        let id = CaptureBoardId::new();
        let board = CaptureBoard_ {
            id,
            name: Some(name.into()),
            inputs: Vec::new(),
            targets: Vec::new(),
            ops: Vec::new(),
        };
        super::storage::register_board(board);
        Self(id)
    }

    pub fn with_target(&self, name: impl Into<String>, tensor: Tensor) -> &Self {
        // Add target to board in storage
        super::storage::add_target_to_board(self.0, name.into(), tensor);
        // Return self for chaining
        self
    }

    pub fn open(&self) {
        super::storage::set_active(Some(self.0));
    }

    pub fn close(&self) {
        super::storage::set_active(None);
    }

    pub fn id(&self) -> CaptureBoardId {
        self.0
    }

    pub fn capture(&self) -> Snapshot {
        let board = super::storage::take_board(self.0).expect("Board not found in storage");

        let reachable = Self::compute_reachable(&board.targets, &board.ops);
        let filtered_ops: Vec<_> = board
            .ops
            .into_iter()
            .filter(|op| reachable.contains(&op.output_id))
            .collect();

        let offset = Self::compute_id_offset(&board.inputs, &board.targets, &filtered_ops);
        let constant_ids = Self::find_constants(&board.inputs, &filtered_ops, &reachable);
        let snapshot_constants = Self::extract_constants(constant_ids, offset);

        let snapshot_inputs = board
            .inputs
            .into_iter()
            .map(|input| SnapshotInput {
                name: input.name,
                id: SnapshotTensorId(input.tensor_id.as_usize() - offset),
                shape: input.shape,
                dtype: input.dtype,
            })
            .collect();

        let snapshot_targets = board
            .targets
            .into_iter()
            .map(|target| SnapshotTarget {
                name: target.name,
                id: SnapshotTensorId(target.tensor_id.as_usize() - offset),
            })
            .collect();

        let snapshot_nodes = filtered_ops
            .into_iter()
            .map(|op| {
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

        Snapshot {
            name: board.name,
            inputs: snapshot_inputs,
            constants: snapshot_constants,
            targets: snapshot_targets,
            nodes: snapshot_nodes,
        }
    }

    fn compute_reachable(targets: &[CapturedTarget], ops: &[CapturedOp]) -> HashSet<TensorId> {
        let mut reachable = HashSet::new();
        let mut to_visit = Vec::new();

        for target in targets {
            to_visit.push(target.tensor_id);
        }

        while let Some(tensor_id) = to_visit.pop() {
            if !reachable.insert(tensor_id) {
                continue;
            }

            for op in ops {
                if op.output_id == tensor_id {
                    for input_id in &op.input_ids {
                        to_visit.push(*input_id);
                    }
                    break;
                }
            }
        }

        reachable
    }

    fn compute_id_offset(inputs: &[CapturedInput], targets: &[CapturedTarget], filtered_ops: &[CapturedOp]) -> usize {
        let mut min_id = usize::MAX;

        for input in inputs {
            min_id = min_id.min(input.tensor_id.as_usize());
        }

        for target in targets {
            min_id = min_id.min(target.tensor_id.as_usize());
        }

        for op in filtered_ops {
            for input_id in &op.input_ids {
                min_id = min_id.min(input_id.as_usize());
            }
            min_id = min_id.min(op.output_id.as_usize());
        }

        if min_id == usize::MAX {
            0
        } else {
            min_id
        }
    }

    fn find_constants(
        inputs: &[CapturedInput],
        filtered_ops: &[CapturedOp],
        reachable: &HashSet<TensorId>,
    ) -> HashSet<TensorId> {
        let input_ids: HashSet<_> = inputs.iter().map(|i| i.tensor_id).collect();
        let op_output_ids: HashSet<_> = filtered_ops.iter().map(|op| op.output_id).collect();

        let mut constant_ids = HashSet::new();
        for op in filtered_ops {
            for &input_id in &op.input_ids {
                if reachable.contains(&input_id) && !input_ids.contains(&input_id) && !op_output_ids.contains(&input_id)
                {
                    constant_ids.insert(input_id);
                }
            }
        }

        constant_ids
    }

    fn extract_constants(constant_ids: HashSet<TensorId>, offset: usize) -> Vec<crate::snapshot::SnapshotConstant> {
        fn to_bytes<T>(vec: Vec<T>) -> Vec<u8> {
            let len = vec.len() * core::mem::size_of::<T>();
            let ptr = vec.as_ptr() as *const u8;
            let mut bytes = Vec::with_capacity(len);
            unsafe {
                core::ptr::copy_nonoverlapping(ptr, bytes.as_mut_ptr(), len);
                bytes.set_len(len);
            }
            core::mem::forget(vec);
            bytes
        }

        constant_ids
            .into_iter()
            .filter_map(|tensor_id| {
                let tensor = crate::tensor::tensor_from_id(tensor_id);
                let dtype = tensor.dtype();
                let shape = tensor.shape();

                let data = match dtype {
                    DType::BOOL => to_bytes(tensor.to_flatten_vec::<bool>().ok()?),
                    DType::F8E4M3 => to_bytes(tensor.to_flatten_vec::<float8::F8E4M3>().ok()?),
                    #[cfg(feature = "f8e5m2")]
                    DType::F8E5M2 => to_bytes(tensor.to_flatten_vec::<float8::F8E5M2>().ok()?),
                    DType::BF16 => to_bytes(tensor.to_flatten_vec::<half::bf16>().ok()?),
                    DType::F16 => to_bytes(tensor.to_flatten_vec::<half::f16>().ok()?),
                    DType::F32 => to_bytes(tensor.to_flatten_vec::<f32>().ok()?),
                    #[cfg(feature = "f64")]
                    DType::F64 => to_bytes(tensor.to_flatten_vec::<f64>().ok()?),
                    DType::U8 => tensor.to_flatten_vec::<u8>().ok()?,
                    #[cfg(feature = "u16")]
                    DType::U16 => to_bytes(tensor.to_flatten_vec::<u16>().ok()?),
                    DType::U32 => to_bytes(tensor.to_flatten_vec::<u32>().ok()?),
                    #[cfg(feature = "u64")]
                    DType::U64 => to_bytes(tensor.to_flatten_vec::<u64>().ok()?),
                    DType::I8 => to_bytes(tensor.to_flatten_vec::<i8>().ok()?),
                    #[cfg(feature = "i16")]
                    DType::I16 => to_bytes(tensor.to_flatten_vec::<i16>().ok()?),
                    DType::I32 => to_bytes(tensor.to_flatten_vec::<i32>().ok()?),
                    #[cfg(feature = "i64")]
                    DType::I64 => to_bytes(tensor.to_flatten_vec::<i64>().ok()?),
                };

                Some(crate::snapshot::SnapshotConstant {
                    id: SnapshotTensorId(tensor_id.as_usize() - offset),
                    name: None,
                    shape,
                    dtype,
                    data,
                })
            })
            .collect()
    }
}

impl Default for CaptureBoard {
    fn default() -> Self {
        Self::new()
    }
}
