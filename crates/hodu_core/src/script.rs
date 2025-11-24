pub mod builder;
pub mod capture;
pub mod snapshot;

#[cfg(feature = "serde")]
use crate::error::{HoduError, HoduResult};
pub use builder::{BuildConfig, BuildType, Builder, TargetArch, TargetConfig, TargetEnv, TargetOS, TargetVendor};
pub use capture::{CaptureBoard, CaptureBoardId, CapturedInput, CapturedOp, CapturedTarget};
pub use snapshot::{Snapshot, SnapshotConstant, SnapshotInput, SnapshotNode, SnapshotTarget, SnapshotTensorId};

/// Script holds the Hodu Script IR and provides compilation/execution interface
pub struct Script {
    snapshot: Snapshot,
}

impl Script {
    /// Create a new Script from a Snapshot
    pub fn new(snapshot: Snapshot) -> Self {
        Self { snapshot }
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

    /// Save Script to a file
    #[cfg(feature = "serde")]
    pub fn save(&self, path: impl AsRef<str>) -> HoduResult<()> {
        let path_str = path.as_ref();

        if !path_str.ends_with(".hdss") {
            return Err(HoduError::InvalidArgument("File must have .hdss extension".into()));
        }

        let serialized = postcard::to_allocvec(&self.snapshot)
            .map_err(|e| HoduError::SerializationFailed(format!("Postcard serialization failed: {}", e)))?;

        #[cfg(feature = "std")]
        {
            std::fs::write(path_str, serialized)
                .map_err(|e| HoduError::IoError(format!("Failed to write file: {}", e)))?;
        }

        #[cfg(not(feature = "std"))]
        {
            return Err(HoduError::UnsupportedOperation(
                "File I/O not available in no_std environment".into(),
            ));
        }

        Ok(())
    }

    /// Load Script from a file
    #[cfg(feature = "serde")]
    pub fn load(path: impl AsRef<str>) -> HoduResult<Self> {
        let path_str = path.as_ref();

        if !path_str.ends_with(".hdss") {
            return Err(HoduError::InvalidArgument("File must have .hdss extension".into()));
        }

        #[cfg(feature = "std")]
        {
            let bytes =
                std::fs::read(path_str).map_err(|e| HoduError::IoError(format!("Failed to read file: {}", e)))?;

            let snapshot: Snapshot = postcard::from_bytes(&bytes)
                .map_err(|e| HoduError::DeserializationFailed(format!("Postcard deserialization failed: {}", e)))?;

            Ok(Self::new(snapshot))
        }

        #[cfg(not(feature = "std"))]
        {
            Err(HoduError::UnsupportedOperation(
                "File I/O not available in no_std environment".into(),
            ))
        }
    }
}
