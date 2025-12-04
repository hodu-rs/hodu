//! Hodu Plugin SDK
//!
//! This crate provides the SDK for developing Hodu plugins using JSON-RPC over stdio.
//!
//! ## Quick Start
//!
//! ```ignore
//! use hodu_plugin_sdk::{
//!     rpc::{LoadModelParams, LoadModelResult, RpcError},
//!     server::PluginServer,
//! };
//!
//! fn main() {
//!     let server = PluginServer::new("my-format", env!("CARGO_PKG_VERSION"))
//!         .model_extensions(vec!["onnx"])
//!         .method("format.load_model", handle_load_model);
//!
//!     server.run().unwrap();
//! }
//!
//! fn handle_load_model(params: LoadModelParams) -> Result<LoadModelResult, RpcError> {
//!     // Implementation
//!     todo!()
//! }
//! ```
//!
//! ## Architecture
//!
//! Plugins are standalone executables that communicate with the CLI via JSON-RPC 2.0 over stdio:
//!
//! ```text
//! CLI <--JSON-RPC 2.0 (stdio)--> Plugin Process
//! ```
//!
//! ## Capabilities
//!
//! Plugins can implement any combination of:
//! - `format.load_model` - Load model from file
//! - `format.save_model` - Save model to file
//! - `format.load_tensor` - Load tensor from file
//! - `format.save_tensor` - Save tensor to file
//! - `backend.run` - Execute model inference
//! - `backend.build` - AOT compile model

mod artifact;
mod backend;
mod build;
mod error;
pub mod rpc;
pub mod server;
mod tensor;

// Re-export notification helpers for convenience
pub use server::{log_debug, log_error, log_info, log_warn, notify_log, notify_progress};

// SDK types
pub use artifact::*;
pub use backend::*;
pub use error::{PluginError, PluginResult};
pub use tensor::*;

// Re-export build templates for hodu CLI
pub use build::{
    cargo_toml_template, main_rs_backend_template, main_rs_model_format_template, main_rs_tensor_format_template,
    manifest_json_backend_template, manifest_json_model_format_template, manifest_json_tensor_format_template,
};

// Re-export from hodu_core for plugin development
pub use hodu_core::{
    format::{hdss, hdt, json},
    op_params::{self, OpParams},
    ops,
    scalar::Scalar,
    snapshot::{self, Snapshot, SnapshotNode},
    tensor::Tensor,
    types::{DType, Device as CoreDevice, Layout, Shape},
};

/// SDK version for compatibility checking (semver string)
pub const SDK_VERSION: &str = env!("CARGO_PKG_VERSION");
