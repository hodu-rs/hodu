//! Hodu Plugin System (std only)
//!
//! This crate provides the plugin system for Hodu:
//! - `BackendPlugin`: Compilers and runtimes (LLVM, Metal, XLA, etc.)
//! - `FormatPlugin`: Model format loaders (ONNX, SafeTensors, etc.)
//! - `PluginManager`: Dynamic plugin loading and management

mod backend;
mod format;
mod manager;
mod output;

pub use backend::*;
pub use format::*;
pub use manager::*;
pub use output::*;

// Re-export from hodu_core
pub use hodu_core::error::{HoduError, HoduResult};
pub use hodu_core::script::{Script, Snapshot};
pub use hodu_core::tensor::Tensor;
pub use hodu_core::types::{DType, Device, Layout, Shape};
