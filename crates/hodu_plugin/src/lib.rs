//! Hodu Plugin System (std only)
//!
//! This crate provides the plugin system for Hodu:
//! - `CompilerPlugin`: Compilers for AOT/JIT compilation (CPU, Metal, LLVM, etc.)
//! - `RuntimePlugin`: Runtimes for executing compiled artifacts (CPU, Metal, CUDA, etc.)
//! - `PluginManager`: Dynamic plugin loading and management
//!
//! ## Architecture
//!
//! ```text
//! Snapshot (IR)
//!     │
//!     ▼
//! ┌─────────────────┐
//! │ CompilerPlugin  │  compile() / build()
//! └─────────────────┘
//!     │
//!     ▼
//! CompiledArtifact
//!     │
//!     ▼
//! ┌─────────────────┐
//! │ RuntimePlugin   │  load() / execute()
//! └─────────────────┘
//!     │
//!     ▼
//! Output Tensors
//! ```

mod artifact;
mod compiler;
mod interp;
mod manager;
mod runtime;

pub use artifact::*;
pub use compiler::*;
pub use interp::*;
pub use manager::*;
pub use runtime::*;

// Re-export from hodu_core
pub use hodu_core::{
    error::{HoduError, HoduResult},
    format::{ModelFormat, OutputFormat, TensorFormat},
    op_metadatas, op_params, ops, snapshot,
    snapshot::Snapshot,
    tensor::Tensor,
    types,
    types::{DType, Device, Layout, Shape},
};
