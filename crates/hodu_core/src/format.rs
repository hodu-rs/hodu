//! Model and tensor format support
//!
//! ## Graph formats (model serialization)
//! - **hdss**: Hodu Snapshot format (native serialized computation graph)
//!
//! ## Tensor formats (input/output data)
//! - **hdt**: Hodu Tensor format (native binary tensor)
//! - **json**: JSON tensor format (human-readable, debugging)

#[cfg(feature = "serde")]
pub mod hdss;
#[cfg(feature = "serde")]
pub mod hdt;
#[cfg(feature = "serde")]
pub mod json;
