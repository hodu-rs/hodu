//! Hodu Snapshot (.hdss) format support

use crate::error::HoduResult;
use crate::snapshot::Snapshot;

pub fn load(path: impl AsRef<std::path::Path>) -> HoduResult<Snapshot> {
    Snapshot::load(path)
}

pub fn save(snapshot: &Snapshot, path: impl AsRef<std::path::Path>) -> HoduResult<()> {
    snapshot.save(path)
}

pub fn to_bytes(snapshot: &Snapshot) -> HoduResult<Vec<u8>> {
    snapshot.to_bytes()
}

pub fn from_bytes(data: &[u8]) -> HoduResult<Snapshot> {
    Snapshot::from_bytes(data)
}
