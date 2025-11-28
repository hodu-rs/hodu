pub mod capture;
pub mod snapshot;

use crate::compat::*;
#[cfg(all(feature = "serde", feature = "std"))]
use crate::error::{HoduError, HoduResult};
pub use capture::{CaptureBoard, CaptureBoardId, CapturedInput, CapturedOp, CapturedTarget};
pub use snapshot::{Snapshot, SnapshotConstant, SnapshotInput, SnapshotNode, SnapshotTarget, SnapshotTensorId};

/// Script holds the Hodu Script IR (Snapshot)
///
/// This is a pure IR container. Compilation and execution are handled by
/// backend plugins in the `hodu_plugin` crate.
pub struct Script {
    snapshot: Snapshot,
    name: Option<String>,
}

impl Script {
    /// Create a new Script from a Snapshot
    pub fn new(snapshot: Snapshot) -> Self {
        Self { snapshot, name: None }
    }

    /// Create a new Script with a name
    pub fn with_name(snapshot: Snapshot, name: impl Into<String>) -> Self {
        Self {
            snapshot,
            name: Some(name.into()),
        }
    }

    /// Get the script name
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set the script name
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Get reference to the underlying snapshot
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    /// Get mutable reference to the underlying snapshot
    pub fn snapshot_mut(&mut self) -> &mut Snapshot {
        &mut self.snapshot
    }

    /// Consume and return the underlying snapshot
    pub fn into_snapshot(self) -> Snapshot {
        self.snapshot
    }

    /// Save Script to a file (.hdss format)
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn save(&self, path: impl AsRef<str>) -> HoduResult<()> {
        let path_str = path.as_ref();

        if !path_str.ends_with(".hdss") {
            return Err(HoduError::InvalidArgument("File must have .hdss extension".into()));
        }

        let serialized = postcard::to_allocvec(&self.snapshot)
            .map_err(|e| HoduError::SerializationFailed(format!("Postcard serialization failed: {}", e)))?;

        std::fs::write(path_str, serialized).map_err(|e| HoduError::IoError(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Load Script from a file (.hdss format)
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn load(path: impl AsRef<str>) -> HoduResult<Self> {
        let path_str = path.as_ref();

        if !path_str.ends_with(".hdss") {
            return Err(HoduError::InvalidArgument("File must have .hdss extension".into()));
        }

        let bytes = std::fs::read(path_str).map_err(|e| HoduError::IoError(format!("Failed to read file: {}", e)))?;

        let snapshot: Snapshot = postcard::from_bytes(&bytes)
            .map_err(|e| HoduError::DeserializationFailed(format!("Postcard deserialization failed: {}", e)))?;

        Ok(Self::new(snapshot))
    }
}
