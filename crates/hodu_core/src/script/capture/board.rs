use crate::{
    compat::*,
    error::HoduResult,
    ops::{Op, OpParams},
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
    ops: Vec<CapturedOp>,
}

impl CaptureBoard {
    pub fn new() -> Self {
        Self {
            id: CaptureBoardId::new(),
            name: None,
            inputs: Vec::new(),
            ops: Vec::new(),
        }
    }

    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            id: CaptureBoardId::new(),
            name: Some(name.into()),
            inputs: Vec::new(),
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

    pub fn ops(&self) -> &[CapturedOp] {
        &self.ops
    }

    pub fn into_parts(self) -> (Vec<CapturedInput>, Vec<CapturedOp>) {
        (self.inputs, self.ops)
    }

    pub fn add_input(&mut self, name: &str, tensor: Tensor) -> HoduResult<()> {
        let input = CapturedInput {
            name: name.to_string(),
            tensor_id: tensor.id(),
            shape: tensor.shape(),
            dtype: tensor.dtype(),
        };
        self.inputs.push(input);
        Ok(())
    }

    pub(super) fn add_op(&mut self, op: CapturedOp) {
        self.ops.push(op);
    }
}

impl Default for CaptureBoard {
    fn default() -> Self {
        Self::new()
    }
}
