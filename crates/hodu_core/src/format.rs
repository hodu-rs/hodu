//! Model and tensor format support
//!
//! ## Graph formats (model serialization)
//! - **hdss**: Hodu Snapshot format (native serialized computation graph)
//!
//! ## Tensor formats (input/output data)
//! - **hdt**: Hodu Tensor format (native binary tensor)
//! - **json**: JSON tensor format (human-readable, debugging)

#![cfg_attr(not(feature = "std"), allow(unused_imports))]

#[allow(clippy::module_inception)]
mod format;
pub mod hdss;
pub mod hdt;
#[cfg(feature = "json")]
pub mod json;

pub use format::*;
