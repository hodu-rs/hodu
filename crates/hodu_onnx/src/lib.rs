//! ONNX import/export for Hodu

mod export;
mod import;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

use hodu_core::error::{HoduError, HoduResult};
use hodu_core::snapshot::Snapshot;
use prost::Message;
use std::path::Path;

/// Load ONNX model from file
pub fn load(path: impl AsRef<Path>) -> HoduResult<Snapshot> {
    let bytes = std::fs::read(path.as_ref()).map_err(|e| HoduError::IoError(format!("Failed to read ONNX: {}", e)))?;
    load_from_bytes(&bytes)
}

/// Load ONNX model from bytes
pub fn load_from_bytes(bytes: &[u8]) -> HoduResult<Snapshot> {
    let model = onnx::ModelProto::decode(bytes)
        .map_err(|e| HoduError::DeserializationFailed(format!("Failed to parse ONNX: {}", e)))?;
    import::import_model(&model)
}

/// Save Snapshot to ONNX file
pub fn save(snapshot: &Snapshot, path: impl AsRef<Path>) -> HoduResult<()> {
    let bytes = save_to_bytes(snapshot)?;
    std::fs::write(path.as_ref(), bytes).map_err(|e| HoduError::IoError(format!("Failed to write ONNX: {}", e)))
}

/// Save Snapshot to ONNX bytes
pub fn save_to_bytes(snapshot: &Snapshot) -> HoduResult<Vec<u8>> {
    let model = export::export_model(snapshot)?;
    let mut bytes = Vec::new();
    model
        .encode(&mut bytes)
        .map_err(|e| HoduError::SerializationFailed(format!("Failed to encode ONNX: {}", e)))?;
    Ok(bytes)
}
