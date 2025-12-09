//! Project templates for creating new plugins
//!
//! These templates are used by `hodu plugin create` command.

/// Plugin protocol version for templates
pub use hodu_plugin::PLUGIN_VERSION;

/// Template for manifest.json (backend plugin)
pub fn manifest_json_backend_template(name: &str) -> String {
    format!(
        r#"{{
  "name": "{name}",
  "version": "0.1.0",
  "description": "Backend plugin for Hodu",
  "license": "MIT",
  "plugin_version": "{PLUGIN_VERSION}",
  "capabilities": ["backend.run"],
  "devices": ["cpu"],
  "dependencies": []
}}
"#
    )
}

/// Template for manifest.json (model format plugin)
pub fn manifest_json_model_format_template(name: &str) -> String {
    format!(
        r#"{{
  "name": "{name}",
  "version": "0.1.0",
  "description": "Model format plugin for Hodu",
  "license": "MIT",
  "plugin_version": "{PLUGIN_VERSION}",
  "capabilities": ["format.load_model"],
  "extensions": ["ext"],
  "dependencies": []
}}
"#
    )
}

/// Template for manifest.json (tensor format plugin)
pub fn manifest_json_tensor_format_template(name: &str) -> String {
    format!(
        r#"{{
  "name": "{name}",
  "version": "0.1.0",
  "description": "Tensor format plugin for Hodu",
  "license": "MIT",
  "plugin_version": "{PLUGIN_VERSION}",
  "capabilities": ["format.load_tensor"],
  "extensions": ["ext"],
  "dependencies": []
}}
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

[[bin]]
name = "{name}"
path = "src/main.rs"

[dependencies]
hodu-plugin-sdk = "{PLUGIN_VERSION}"

# For backend plugins that need CPU kernels:
# hodu_cpu_kernels = "0.2"
"#
    )
}

/// Template for main.rs (backend plugin)
pub fn main_rs_backend_template(name: &str) -> String {
    format!(
        r#"//! {name} - Backend plugin for Hodu
//!
//! This plugin implements model execution for the specified device.

use hodu_plugin_sdk::{{
    rpc::{{RunParams, RunResult, RpcError, TensorOutput}},
    server::PluginServer,
    hdss, TensorData,
}};
use std::collections::HashMap;

fn main() {{
    let server = PluginServer::new("{name}", env!("CARGO_PKG_VERSION"))
        .devices(vec!["cpu"])  // TODO: Update with your supported devices
        .method("backend.run", handle_run);

    if let Err(e) = server.run() {{
        eprintln!("Plugin error: {{}}", e);
        std::process::exit(1);
    }}
}}

fn handle_run(params: RunParams) -> Result<RunResult, RpcError> {{
    // Load the model snapshot
    let snapshot = hdss::load(&params.snapshot_path)
        .map_err(|e| RpcError::internal_error(format!("Failed to load snapshot: {{}}", e)))?;

    // Load input tensors
    let mut inputs: HashMap<String, TensorData> = HashMap::new();
    for input in &params.inputs {{
        let tensor = TensorData::load(&input.path)
            .map_err(|e| RpcError::internal_error(format!("Failed to load input '{{}}': {{}}", input.name, e)))?;
        inputs.insert(input.name.clone(), tensor);
    }}

    // TODO: Implement your model execution logic here
    //
    // Example workflow:
    // 1. Parse the snapshot to get the computation graph
    // 2. Execute each operation in topological order
    // 3. Collect output tensors
    //
    // For now, this returns an error indicating not implemented
    Err(RpcError::internal_error(format!(
        "Backend '{{}}' execution not implemented. Model has {{}} nodes, {{}} inputs provided.",
        params.device,
        snapshot.nodes.len(),
        inputs.len()
    )))
}}
"#
    )
}

/// Template for main.rs (model format plugin)
pub fn main_rs_model_format_template(name: &str) -> String {
    format!(
        r#"//! {name} - Model format plugin for Hodu
//!
//! This plugin loads models from a specific format and converts them to Hodu snapshots.

use hodu_plugin_sdk::{{
    rpc::{{LoadModelParams, LoadModelResult, RpcError}},
    server::PluginServer,
    hdss,
    snapshot::{{Snapshot, SnapshotNode}},
}};
use std::path::Path;

fn main() {{
    let server = PluginServer::new("{name}", env!("CARGO_PKG_VERSION"))
        .model_extensions(vec!["ext"])  // TODO: Update with your supported extensions (e.g., "onnx", "pb")
        .method("format.load_model", handle_load_model);

    if let Err(e) = server.run() {{
        eprintln!("Plugin error: {{}}", e);
        std::process::exit(1);
    }}
}}

fn handle_load_model(params: LoadModelParams) -> Result<LoadModelResult, RpcError> {{
    let path = Path::new(&params.path);

    // Check if file exists
    if !path.exists() {{
        return Err(RpcError::invalid_params(format!("File not found: {{}}", params.path)));
    }}

    // TODO: Implement your model parsing logic here
    //
    // Example workflow:
    // 1. Read and parse the model file
    // 2. Convert operations to Hodu snapshot nodes
    // 3. Build the computation graph
    //
    // For now, return an error with helpful information
    let file_size = std::fs::metadata(path)
        .map(|m| m.len())
        .unwrap_or(0);

    Err(RpcError::internal_error(format!(
        "Model format '{{}}' parsing not implemented. File: {{}} ({{}} bytes)",
        path.extension().and_then(|e| e.to_str()).unwrap_or("unknown"),
        params.path,
        file_size
    )))
}}
"#
    )
}

/// Template for main.rs (tensor format plugin)
pub fn main_rs_tensor_format_template(name: &str) -> String {
    format!(
        r#"//! {name} - Tensor format plugin for Hodu
//!
//! This plugin loads tensors from a specific format and converts them to Hodu tensor format.

use hodu_plugin_sdk::{{
    rpc::{{LoadTensorParams, LoadTensorResult, RpcError}},
    server::PluginServer,
    TensorData,
}};
use std::path::Path;

fn main() {{
    let server = PluginServer::new("{name}", env!("CARGO_PKG_VERSION"))
        .tensor_extensions(vec!["ext"])  // TODO: Update with your supported extensions (e.g., "npy", "npz")
        .method("format.load_tensor", handle_load_tensor);

    if let Err(e) = server.run() {{
        eprintln!("Plugin error: {{}}", e);
        std::process::exit(1);
    }}
}}

fn handle_load_tensor(params: LoadTensorParams) -> Result<LoadTensorResult, RpcError> {{
    let path = Path::new(&params.path);

    // Check if file exists
    if !path.exists() {{
        return Err(RpcError::invalid_params(format!("File not found: {{}}", params.path)));
    }}

    // TODO: Implement your tensor parsing logic here
    //
    // Example workflow:
    // 1. Read and parse the tensor file
    // 2. Extract shape, dtype, and data
    // 3. Create TensorData and save as .hdt
    //
    // For now, return an error with helpful information
    let file_size = std::fs::metadata(path)
        .map(|m| m.len())
        .unwrap_or(0);

    Err(RpcError::internal_error(format!(
        "Tensor format '{{}}' parsing not implemented. File: {{}} ({{}} bytes)",
        path.extension().and_then(|e| e.to_str()).unwrap_or("unknown"),
        params.path,
        file_size
    )))
}}
"#
    )
}
