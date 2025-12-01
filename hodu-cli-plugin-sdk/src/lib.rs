//! Hodu CLI Plugin SDK
//!
//! This crate provides the SDK for developing Hodu CLI plugins:
//! - `BackendPlugin`: For model execution (Runner) and AOT compilation (Builder)
//! - `FormatPlugin`: For file format support (ONNX, safetensors, npy, etc.)
//!
//! ## Quick Start
//!
//! 1. Create `info.toml` with plugin metadata
//! 2. Call `hodu_cli_plugin_sdk::build()` in your `build.rs`
//! 3. Use `#[derive(BackendPlugin)]` or `#[derive(FormatPlugin)]`
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
//! use hodu_cli_plugin_sdk::*;
//! use std::collections::HashMap;
//!
//! #[derive(Default, BackendPlugin)]
//! pub struct MyBackend;
//!
//! impl MyBackend {
//!     // Required when runner = true in info.toml
//!     pub fn run(
//!         &self,
//!         snapshot: &Snapshot,
//!         device: Device,
//!         inputs: &[(&str, TensorData)],
//!     ) -> PluginResult<HashMap<String, TensorData>> {
//!         todo!("Implement model execution")
//!     }
//!
//!     // Required when builder = true in info.toml
//!     // pub fn supported_formats(&self, target: &BuildTarget) -> Vec<BuildFormat> { ... }
//!     // pub fn build(&self, snapshot: &Snapshot, target: &BuildTarget, ...) -> PluginResult<()> { ... }
//! }
//! ```
//!
//! ### FormatPlugin
//!
//! Handles file format conversion:
//! - Load/save models (Snapshot)
//! - Load/save tensors (TensorData)
//!
//! ```ignore
//! use hodu_cli_plugin_sdk::*;
//! use std::path::Path;
//!
//! #[derive(Default, FormatPlugin)]
//! pub struct OnnxFormat;
//!
//! impl OnnxFormat {
//!     // Required when load_model = true in info.toml
//!     pub fn load_model(&self, path: &Path) -> PluginResult<Snapshot> {
//!         todo!("Implement model loading")
//!     }
//!
//!     pub fn load_model_from_bytes(&self, data: &[u8]) -> PluginResult<Snapshot> {
//!         todo!("Implement model loading from bytes")
//!     }
//!
//!     // Other methods based on info.toml capabilities...
//! }
//! ```

mod artifact;
mod backend;
mod build;
mod error;
mod format;
pub mod info;
mod tensor;

// SDK types
pub use artifact::*;
pub use backend::*;
pub use build::build;
pub use error::{PluginError, PluginResult};
pub use format::*;
pub use hodu_cli_plugin_sdk_macros::{BackendPlugin, FormatPlugin};
pub use info::{BackendInfo, FormatInfo, PluginInfo, PluginInfoType, PluginMeta};
pub use tensor::*;

// Re-export build templates for hodu CLI
pub use build::{
    build_rs_template, cargo_toml_template, info_toml_backend_template, info_toml_format_template,
    lib_rs_backend_template, lib_rs_format_template,
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

// SDK-specific DType for ABI stability (re-exported from tensor.rs)
pub use tensor::SdkDType;

/// SDK version for compatibility checking (semver string)
pub const SDK_VERSION: &str = env!("CARGO_PKG_VERSION");

/// FFI protocol version for ABI compatibility checking
///
/// This version is bumped when there are breaking changes to the FFI interface
/// (e.g., struct layout changes, function signature changes).
///
/// - Version 1: Initial stable FFI protocol
///
/// The CLI will reject plugins with a different FFI protocol version.
pub const FFI_PROTOCOL_VERSION: u32 = 1;

// Helper functions for proc-macro generated code (not public API)
#[doc(hidden)]
pub const fn __parse_bool(s: &str) -> bool {
    matches!(s.as_bytes(), b"true" | b"True" | b"TRUE" | b"1")
}

#[doc(hidden)]
pub fn __parse_devices(s: &str) -> Vec<Device> {
    if s.is_empty() {
        return vec![];
    }
    s.split(',').filter_map(|d| Device::parse(d.trim())).collect()
}

#[doc(hidden)]
pub fn __parse_targets(s: &str) -> Vec<BuildTarget> {
    if s.is_empty() {
        return vec![];
    }
    // Format: "triple@device,triple@device" e.g., "x86_64-linux-gnu@CPU,aarch64-apple-darwin@Metal"
    s.split(',')
        .filter_map(|t| {
            let t = t.trim();
            if let Some((triple, device_str)) = t.split_once('@') {
                Device::parse(device_str).map(|device| BuildTarget::new(triple, device))
            } else {
                // Default to CPU if no device specified
                Some(BuildTarget::new(t, Device::CPU))
            }
        })
        .collect()
}

#[doc(hidden)]
pub fn __parse_extensions(s: &'static str) -> Vec<&'static str> {
    if s.is_empty() {
        return vec![];
    }
    s.split(',').map(|e| e.trim()).collect()
}
