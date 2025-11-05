use super::builder::ir::Module;
use crate::{error::HoduResult, layer::compat::*};

/// Serialize a module to bytes (no_std compatible)
#[cfg(feature = "serde")]
pub(crate) fn save_module_to_bytes(module: &Module) -> HoduResult<Vec<u8>> {
    let config = bincode::config::standard();
    let bytes = bincode::encode_to_vec(module, config)?;
    Ok(bytes)
}

/// Deserialize a module from bytes (no_std compatible)
#[cfg(feature = "serde")]
pub(crate) fn load_module_from_bytes(bytes: &[u8]) -> HoduResult<Module> {
    let config = bincode::config::standard();
    let (module, _len) = bincode::decode_from_slice(bytes, config)?;
    Ok(module)
}

/// Save a module to file (requires std)
#[cfg(all(feature = "serde", feature = "std"))]
pub(crate) fn save_module<P: AsRef<std::path::Path>>(module: &Module, path: P) -> HoduResult<()> {
    let bytes = save_module_to_bytes(module)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Load a module from file (requires std)
#[cfg(all(feature = "serde", feature = "std"))]
pub(crate) fn load_module<P: AsRef<std::path::Path>>(path: P) -> HoduResult<Module> {
    let bytes = std::fs::read(path)?;
    load_module_from_bytes(&bytes)
}
