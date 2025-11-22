use crate::{
    compat::*,
    ops::{Op, OpParams},
    types::{DType, Layout, Shape},
};

/// Snapshot-local tensor ID (normalized from runtime TensorId)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SnapshotTensorId(pub usize);

/// Snapshot input specification
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SnapshotInput {
    pub name: String,
    pub id: SnapshotTensorId,
    pub shape: Shape,
    pub dtype: DType,
}

/// Snapshot target (output) specification
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SnapshotTarget {
    pub name: String,
    pub id: SnapshotTensorId,
}

/// Snapshot node (operation)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SnapshotNode {
    pub op: Op,
    pub params: Option<OpParams>,
    pub input_ids: Vec<SnapshotTensorId>,
    pub output_id: SnapshotTensorId,
    pub input_layouts: Vec<Layout>,
    pub output_layout: Layout,
}

/// Hodu Snapshot - serializable IR representation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Snapshot {
    pub name: Option<String>,
    pub inputs: Vec<SnapshotInput>,
    pub targets: Vec<SnapshotTarget>,
    pub nodes: Vec<SnapshotNode>,
}

impl Snapshot {
    pub fn new() -> Self {
        Self {
            name: None,
            inputs: Vec::new(),
            targets: Vec::new(),
            nodes: Vec::new(),
        }
    }

    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            inputs: Vec::new(),
            targets: Vec::new(),
            nodes: Vec::new(),
        }
    }
}

impl Default for Snapshot {
    fn default() -> Self {
        Self::new()
    }
}
