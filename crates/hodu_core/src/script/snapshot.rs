use crate::{
    compat::*,
    ops::{Op, OpParams},
    types::{DType, Layout, Shape},
};

/// Snapshot-local tensor ID (normalized from runtime TensorId)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

/// Snapshot constant tensor (weights, biases, etc.)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SnapshotConstant {
    pub id: SnapshotTensorId,
    pub shape: Shape,
    pub dtype: DType,
    /// Raw tensor data in bytes
    pub data: Vec<u8>,
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
    pub output_dtype: DType,
}

/// Hodu Snapshot - serializable IR representation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Snapshot {
    pub name: Option<String>,
    pub inputs: Vec<SnapshotInput>,
    pub constants: Vec<SnapshotConstant>,
    pub targets: Vec<SnapshotTarget>,
    pub nodes: Vec<SnapshotNode>,
}

impl Snapshot {
    pub fn new() -> Self {
        Self {
            name: None,
            inputs: Vec::new(),
            constants: Vec::new(),
            targets: Vec::new(),
            nodes: Vec::new(),
        }
    }

    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            inputs: Vec::new(),
            constants: Vec::new(),
            targets: Vec::new(),
            nodes: Vec::new(),
        }
    }

    /// Serialize snapshot to bytes
    #[cfg(feature = "serde")]
    pub fn serialize(&self) -> crate::error::HoduResult<Vec<u8>> {
        postcard::to_allocvec(self)
            .map_err(|e| crate::error::HoduError::SerializationFailed(format!("Failed to serialize Snapshot: {}", e)))
    }

    /// Deserialize snapshot from bytes
    #[cfg(feature = "serde")]
    pub fn deserialize(data: &[u8]) -> crate::error::HoduResult<Self> {
        postcard::from_bytes(data).map_err(|e| {
            crate::error::HoduError::DeserializationFailed(format!("Failed to deserialize Snapshot: {}", e))
        })
    }
}

impl Default for Snapshot {
    fn default() -> Self {
        Self::new()
    }
}
