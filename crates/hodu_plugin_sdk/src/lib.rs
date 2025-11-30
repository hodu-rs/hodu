//! Hodu Plugin SDK
//!
//! This crate provides the SDK for developing Hodu plugins:
//! - `BackendPlugin`: For model execution (Runner) and AOT compilation (Builder)
//! - `FormatPlugin`: For file format support (ONNX, safetensors, npy, etc.)
//!
//! ## Plugin Types
//!
//! ### BackendPlugin
//!
//! Combines Runner and Builder capabilities:
//! - **Runner**: Execute models on a device (`hodu run`)
//! - **Builder**: AOT compile models to native artifacts (`hodu build`)
//!
//! ```ignore
//! use hodu_plugin_sdk::*;
//!
//! #[derive(Default)]
//! pub struct MyBackend;
//!
//! impl BackendPlugin for MyBackend {
//!     fn name(&self) -> &str { "my-backend" }
//!     fn version(&self) -> &str { "0.1.0" }
//!     fn capabilities(&self) -> BackendCapabilities { BackendCapabilities::runner_only() }
//!     // ...
//! }
//!
//! export_backend_plugin!(MyBackend);
//! ```
//!
//! ### FormatPlugin
//!
//! Handles file format conversion:
//! - Load/save models (Snapshot)
//! - Load/save tensors (TensorData)
//!
//! ```ignore
//! use hodu_plugin_sdk::*;
//!
//! #[derive(Default)]
//! pub struct NpyFormat;
//!
//! impl FormatPlugin for NpyFormat {
//!     fn name(&self) -> &str { "npy" }
//!     fn version(&self) -> &str { "0.1.0" }
//!     fn capabilities(&self) -> FormatCapabilities { FormatCapabilities::tensor_only() }
//!     fn supported_extensions(&self) -> Vec<&str> { vec!["npy", "npz"] }
//!     // ...
//! }
//!
//! export_format_plugin!(NpyFormat);
//! ```

mod artifact;
mod backend;
mod format;
mod tensor;

// SDK types
pub use artifact::*;
pub use backend::*;
pub use format::*;
pub use tensor::*;

// Re-export from hodu_core for plugin development
pub use hodu_core::{
    error::{HoduError, HoduResult},
    op_params::{self, OpParams},
    ops,
    scalar::Scalar,
    snapshot::{self, Snapshot, SnapshotNode},
    types::{DType, Device, Layout, Shape},
};

/// SDK version for compatibility checking
pub const SDK_VERSION: &str = env!("CARGO_PKG_VERSION");
