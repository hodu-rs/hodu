//! Hodu Plugin System (std only)
//!
//! This crate provides the plugin system for Hodu:
//! - `CompilerPlugin`: Compilers for AOT/JIT compilation (LLVM, Metal, XLA, etc.)
//! - `RuntimePlugin`: Runtimes for executing compiled artifacts (Native, CUDA, Metal, etc.)
//! - `FormatPlugin`: Model format loaders (ONNX, SafeTensors, etc.)
//! - `PluginManager`: Dynamic plugin loading and management
//!
//! ## Architecture
//!
//! ```text
//! Script (IR)
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
mod format;
mod interp;
mod manager;
mod output;
mod runtime;

pub use artifact::*;
pub use compiler::*;
pub use format::*;
pub use interp::*;
pub use manager::*;
pub use output::*;
pub use runtime::*;

// Re-export from hodu_core
pub use hodu_core::error::{HoduError, HoduResult};
pub use hodu_core::script::{Script, Snapshot};
pub use hodu_core::tensor::Tensor;
pub use hodu_core::types::{DType, Device, Layout, Shape};
