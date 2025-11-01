//! Prelude module for convenient imports
//!
//! Usage: `use hodu_core::prelude::*;`

// Re-export
pub use crate::error::HoduResult;
pub use crate::tensor::{get_runtime_device, set_runtime_device, Tensor};
pub use crate::types::*;
