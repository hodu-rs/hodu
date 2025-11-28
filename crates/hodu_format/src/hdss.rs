//! Hodu Snapshot (.hdss) format support
//!
//! The hdss format is the native serialized format for Hodu computation graphs.
//! It stores the complete Snapshot IR which can be loaded and executed by the
//! InterpRuntime without any compilation.

use hodu_core::error::{HoduError, HoduResult};
use hodu_core::script::Snapshot;

/// Load a Snapshot from hdss file
#[cfg(feature = "std")]
pub fn load(path: impl AsRef<std::path::Path>) -> HoduResult<Snapshot> {
    let data =
        std::fs::read(path.as_ref()).map_err(|e| HoduError::IoError(format!("Failed to read hdss file: {}", e)))?;
    Snapshot::deserialize(&data)
}

/// Save a Snapshot to hdss file
#[cfg(feature = "std")]
pub fn save(snapshot: &Snapshot, path: impl AsRef<std::path::Path>) -> HoduResult<()> {
    let data = snapshot.serialize()?;
    std::fs::write(path.as_ref(), data).map_err(|e| HoduError::IoError(format!("Failed to write hdss file: {}", e)))?;
    Ok(())
}

/// Serialize a Snapshot to bytes
pub fn serialize(snapshot: &Snapshot) -> HoduResult<Vec<u8>> {
    snapshot.serialize()
}

/// Deserialize a Snapshot from bytes
pub fn deserialize(data: &[u8]) -> HoduResult<Snapshot> {
    Snapshot::deserialize(data)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_empty_snapshot() {
        let snapshot = Snapshot::new();
        let data = serialize(&snapshot).unwrap();
        let restored = deserialize(&data).unwrap();
        assert_eq!(snapshot.inputs.len(), restored.inputs.len());
        assert_eq!(snapshot.nodes.len(), restored.nodes.len());
    }
}
