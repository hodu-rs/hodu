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
pub mod server;
mod tensor;

// Re-export rpc module from hodu_plugin
pub use hodu_plugin::rpc;

// Re-export from hodu_plugin (common types shared with hodu-cli)
pub use hodu_plugin::{BuildTarget, Device, PluginDType, PluginError, PluginResult, TensorData, PLUGIN_VERSION};

// Re-export backend utilities from hodu_plugin
pub use hodu_plugin::{current_host_triple, device_type, parse_device_id};

// Re-export tensor types with hodu_core extensions
pub use tensor::{SdkDType, TensorDataExt};

// Re-export notification helpers for convenience
pub use server::{log_debug, log_error, log_info, log_warn, notify_log, notify_progress};

// Plugin SDK specific types (for plugin development only)
pub use artifact::*;
pub use backend::{
    check_build_capability, host_matches_pattern, is_tool_available, BuildCapability, PluginManifest, SupportedTarget,
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
