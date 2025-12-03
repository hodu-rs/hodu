//! Build script support for Hodu plugins
//!
//! This module provides the `build()` function for plugin build.rs files.
//!
//! # Usage
//!
//! In your plugin's `build.rs`:
//!
//! ```ignore
//! fn main() {
//!     hodu_plugin_sdk::build();
//! }
//! ```
//!
//! And in `Cargo.toml`:
//!
//! ```toml
//! [build-dependencies]
//! hodu_plugin_sdk = "0.3"
//! ```

use crate::info::{PluginInfo, PluginInfoType};

/// Build function for plugin build.rs
///
/// This function:
/// 1. Reads info.toml and parses plugin metadata
/// 2. Sets environment variables for the derive macros
///
/// # Environment Variables Set
///
/// Common:
/// - `HODU_PLUGIN_NAME`: Plugin name
/// - `HODU_PLUGIN_VERSION`: Plugin version
/// - `HODU_PLUGIN_TYPE`: "backend" or "format"
///
/// For backend plugins:
/// - `HODU_BACKEND_RUNNER`: "true" or "false"
/// - `HODU_BACKEND_BUILDER`: "true" or "false"
/// - `HODU_BACKEND_DEVICES`: Comma-separated list (e.g., "CPU,CUDA")
/// - `HODU_BACKEND_TARGETS`: Comma-separated list
///
/// For format plugins:
/// - `HODU_FORMAT_LOAD_MODEL`: "true" or "false"
/// - `HODU_FORMAT_SAVE_MODEL`: "true" or "false"
/// - `HODU_FORMAT_LOAD_TENSOR`: "true" or "false"
/// - `HODU_FORMAT_SAVE_TENSOR`: "true" or "false"
/// - `HODU_FORMAT_EXTENSIONS`: Comma-separated list (e.g., "onnx,pb")
///
/// # Panics
///
/// Panics if info.toml is missing or malformed.
pub fn build() {
    println!("cargo::rerun-if-changed=info.toml");

    // Read and parse info.toml
    let info_content =
        std::fs::read_to_string("info.toml").expect("Failed to read info.toml. Make sure it exists in the crate root.");

    let info = PluginInfo::parse(&info_content).expect("Failed to parse info.toml");

    info.validate().expect("Invalid info.toml");

    // Common fields
    println!("cargo::rustc-env=HODU_PLUGIN_NAME={}", info.plugin.name);
    println!("cargo::rustc-env=HODU_PLUGIN_VERSION={}", info.plugin.version);

    match info.plugin.plugin_type {
        PluginInfoType::Backend => {
            println!("cargo::rustc-env=HODU_PLUGIN_TYPE=backend");

            let backend = info.backend.unwrap();
            println!("cargo::rustc-env=HODU_BACKEND_RUNNER={}", backend.runner);
            println!("cargo::rustc-env=HODU_BACKEND_BUILDER={}", backend.builder);
            println!("cargo::rustc-env=HODU_BACKEND_DEVICES={}", backend.devices.join(","));
            println!("cargo::rustc-env=HODU_BACKEND_TARGETS={}", backend.targets.join(","));
        },
        PluginInfoType::Format => {
            println!("cargo::rustc-env=HODU_PLUGIN_TYPE=format");

            let format = info.format.unwrap();
            println!("cargo::rustc-env=HODU_FORMAT_LOAD_MODEL={}", format.load_model);
            println!("cargo::rustc-env=HODU_FORMAT_SAVE_MODEL={}", format.save_model);
            println!("cargo::rustc-env=HODU_FORMAT_LOAD_TENSOR={}", format.load_tensor);
            println!("cargo::rustc-env=HODU_FORMAT_SAVE_TENSOR={}", format.save_tensor);
            println!(
                "cargo::rustc-env=HODU_FORMAT_EXTENSIONS={}",
                format.extensions.join(",")
            );
        },
    }
}

/// Template for info.toml (backend plugin)
pub fn info_toml_backend_template(name: &str, version: &str) -> String {
    format!(
        r#"[plugin]
name = "{name}"
version = "{version}"
type = "backend"

[backend]
runner = true
builder = false
devices = ["CPU"]
targets = []
"#
    )
}

/// Template for info.toml (format plugin)
pub fn info_toml_format_template(name: &str, version: &str) -> String {
    format!(
        r#"[plugin]
name = "{name}"
version = "{version}"
type = "format"

[format]
load_model = true
save_model = false
load_tensor = false
save_tensor = false
extensions = []
"#
    )
}

/// Template for Cargo.toml
pub fn cargo_toml_template(name: &str) -> String {
    format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[build-dependencies]
hodu_plugin_sdk = "0.3"

[dependencies]
hodu_plugin_sdk = "0.3"

# For backend plugins that need CPU kernels:
# hodu_cpu_kernels = "0.2"
"#
    )
}

/// Template for build.rs
pub fn build_rs_template() -> &'static str {
    r#"fn main() {
    hodu_plugin_sdk::build();
}
"#
}

/// Template for lib.rs (backend plugin)
pub fn lib_rs_backend_template(struct_name: &str) -> String {
    format!(
        r#"use hodu_plugin_sdk::{{
    BackendPlugin, BuildFormat, BuildTarget, Device, PluginResult, Snapshot, TensorData,
}};
use std::collections::HashMap;
use std::path::Path;

#[derive(Default, BackendPlugin)]
pub struct {struct_name};

impl {struct_name} {{
    /// Execute the model on the given device
    ///
    /// Required when `runner = true` in info.toml.
    pub fn run(
        &self,
        _snapshot: &Snapshot,
        _device: Device,
        _inputs: &[(&str, TensorData)],
    ) -> PluginResult<HashMap<String, TensorData>> {{
        todo!("Implement model execution")
    }}

    // === Builder methods (required when `builder = true` in info.toml) ===

    // /// Return supported output formats for a given target
    // pub fn supported_formats(&self, _target: &BuildTarget) -> Vec<BuildFormat> {{
    //     vec![BuildFormat::Object, BuildFormat::StaticLib]
    // }}

    // /// Build an AOT artifact
    // pub fn build(
    //     &self,
    //     _snapshot: &Snapshot,
    //     _target: &BuildTarget,
    //     _format: BuildFormat,
    //     _output: &Path,
    // ) -> PluginResult<()> {{
    //     todo!("Implement AOT compilation")
    // }}
}}
"#
    )
}

/// Template for lib.rs (format plugin)
pub fn lib_rs_format_template(struct_name: &str) -> String {
    format!(
        r#"use hodu_plugin_sdk::{{
    FormatPlugin, PluginResult, Snapshot, TensorData,
}};
use std::path::Path;

#[derive(Default, FormatPlugin)]
pub struct {struct_name};

impl {struct_name} {{
    // === Model loading (required when `load_model = true` in info.toml) ===

    /// Load a model from a file path
    pub fn load_model(&self, _path: &Path) -> PluginResult<Snapshot> {{
        todo!("Implement model loading")
    }}

    /// Load a model from bytes
    pub fn load_model_from_bytes(&self, _data: &[u8]) -> PluginResult<Snapshot> {{
        todo!("Implement model loading from bytes")
    }}

    // === Model saving (required when `save_model = true` in info.toml) ===

    // pub fn save_model(&self, _snapshot: &Snapshot, _path: &Path) -> PluginResult<()> {{
    //     todo!("Implement model saving")
    // }}

    // pub fn save_model_to_bytes(&self, _snapshot: &Snapshot) -> PluginResult<Vec<u8>> {{
    //     todo!("Implement model saving to bytes")
    // }}

    // === Tensor loading (required when `load_tensor = true` in info.toml) ===

    // pub fn load_tensor(&self, _path: &Path) -> PluginResult<TensorData> {{
    //     todo!("Implement tensor loading")
    // }}

    // pub fn load_tensor_from_bytes(&self, _data: &[u8]) -> PluginResult<TensorData> {{
    //     todo!("Implement tensor loading from bytes")
    // }}

    // === Tensor saving (required when `save_tensor = true` in info.toml) ===

    // pub fn save_tensor(&self, _tensor: &TensorData, _path: &Path) -> PluginResult<()> {{
    //     todo!("Implement tensor saving")
    // }}

    // pub fn save_tensor_to_bytes(&self, _tensor: &TensorData) -> PluginResult<Vec<u8>> {{
    //     todo!("Implement tensor saving to bytes")
    // }}
}}
"#
    )
}
