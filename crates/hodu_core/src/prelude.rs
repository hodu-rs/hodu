//! Prelude module for convenient imports
//!
//! Usage: `use hodu_core::prelude::*;`

// Re-export core library types and functions
pub use crate::backends::{builder::Builder, script::Script};
pub use crate::error::HoduResult;
pub use crate::scalar::Scalar;
pub use crate::tensor::{set_runtime_device, Tensor};
pub use crate::types::{backend::Backend, device::Device, dtype::DType};
