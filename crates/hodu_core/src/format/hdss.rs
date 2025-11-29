//! Hodu Snapshot (.hdss) format support

use crate::error::HoduResult;
use crate::snapshot::Snapshot;

#[cfg(all(feature = "serde", feature = "std"))]
pub fn load(path: impl AsRef<std::path::Path>) -> HoduResult<Snapshot> {
    Snapshot::load(path)
}

#[cfg(all(feature = "serde", feature = "std"))]
pub fn save(snapshot: &Snapshot, path: impl AsRef<std::path::Path>) -> HoduResult<()> {
    snapshot.save(path)
}

#[cfg(feature = "serde")]
pub fn to_bytes(snapshot: &Snapshot) -> HoduResult<Vec<u8>> {
    snapshot.to_bytes()
}

#[cfg(feature = "serde")]
pub fn from_bytes(data: &[u8]) -> HoduResult<Snapshot> {
    Snapshot::from_bytes(data)
}
