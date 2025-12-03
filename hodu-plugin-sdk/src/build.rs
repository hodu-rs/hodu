//! Project templates for creating new plugins
//!
//! These templates are used by `hodu plugin create` command.

/// Template for Cargo.toml
pub fn cargo_toml_template(name: &str) -> String {
    let sdk_version = env!("CARGO_PKG_VERSION");
    format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "{name}"
path = "src/main.rs"

[dependencies]
hodu_plugin_sdk = "{sdk_version}"

# For backend plugins that need CPU kernels:
# hodu_cpu_kernels = "0.2"
"#
    )
}

/// Template for main.rs (backend plugin)
pub fn main_rs_backend_template(name: &str) -> String {
    format!(
        r#"use hodu_plugin_sdk::{{
    rpc::{{RunParams, RunResult, RpcError, TensorOutput}},
    server::PluginServer,
    hdss, TensorData,
}};
use std::collections::HashMap;

fn main() {{
    let server = PluginServer::new("{name}", env!("CARGO_PKG_VERSION"))
        .devices(vec!["cpu"])  // Update with supported devices
        .method("backend.run", handle_run);

    if let Err(e) = server.run() {{
        eprintln!("Plugin error: {{}}", e);
        std::process::exit(1);
    }}
}}

fn handle_run(params: RunParams) -> Result<RunResult, RpcError> {{
    // Load snapshot
    let snapshot = hdss::load(&params.snapshot_path)
        .map_err(|e| RpcError::internal_error(e.to_string()))?;

    // Load input tensors
    let mut inputs: HashMap<String, TensorData> = HashMap::new();
    for input in &params.inputs {{
        let tensor = TensorData::load(&input.path)
            .map_err(|e| RpcError::internal_error(e.to_string()))?;
        inputs.insert(input.name.clone(), tensor);
    }}

    // TODO: Implement model execution
    // let outputs = execute(&snapshot, &params.device, &inputs)?;

    // For now, return empty outputs
    let outputs: Vec<TensorOutput> = vec![];

    Ok(RunResult {{ outputs }})
}}
"#
    )
}

/// Template for main.rs (model format plugin)
pub fn main_rs_model_format_template(name: &str) -> String {
    format!(
        r#"use hodu_plugin_sdk::{{
    rpc::{{LoadModelParams, LoadModelResult, RpcError}},
    server::PluginServer,
    hdss,
}};
use std::path::Path;

fn main() {{
    let server = PluginServer::new("{name}", env!("CARGO_PKG_VERSION"))
        .extensions(vec!["ext"])  // Update with supported extensions
        .method("format.load_model", handle_load_model);

    if let Err(e) = server.run() {{
        eprintln!("Plugin error: {{}}", e);
        std::process::exit(1);
    }}
}}

fn handle_load_model(params: LoadModelParams) -> Result<LoadModelResult, RpcError> {{
    let _path = Path::new(&params.path);

    // TODO: Implement model loading
    // Parse the model file and convert to Hodu Snapshot
    // let snapshot = parse_model(path)?;

    // For now, return error
    Err(RpcError::internal_error("Not implemented"))

    // Save snapshot to temp file and return path
    // let snapshot_path = format!("/tmp/snapshot_{{}}.hdss", uuid::Uuid::new_v4());
    // hdss::save(&snapshot, &snapshot_path)
    //     .map_err(|e| RpcError::internal_error(e.to_string()))?;
    //
    // Ok(LoadModelResult {{ snapshot_path }})
}}
"#
    )
}

/// Template for main.rs (tensor format plugin)
pub fn main_rs_tensor_format_template(name: &str) -> String {
    format!(
        r#"use hodu_plugin_sdk::{{
    rpc::{{LoadTensorParams, LoadTensorResult, RpcError}},
    server::PluginServer,
    TensorData,
}};
use std::path::Path;

fn main() {{
    let server = PluginServer::new("{name}", env!("CARGO_PKG_VERSION"))
        .extensions(vec!["ext"])  // Update with supported extensions
        .method("format.load_tensor", handle_load_tensor);

    if let Err(e) = server.run() {{
        eprintln!("Plugin error: {{}}", e);
        std::process::exit(1);
    }}
}}

fn handle_load_tensor(params: LoadTensorParams) -> Result<LoadTensorResult, RpcError> {{
    let _path = Path::new(&params.path);

    // TODO: Implement tensor loading
    // Parse the tensor file and save as .hdt format
    // let tensor = parse_tensor(path)?;

    // For now, return error
    Err(RpcError::internal_error("Not implemented"))

    // Save tensor to temp file and return path
    // let tensor_path = format!("/tmp/tensor_{{}}.hdt", std::process::id());
    // tensor.save(&tensor_path)
    //     .map_err(|e| RpcError::internal_error(e.to_string()))?;
    //
    // Ok(LoadTensorResult {{ tensor_path }})
}}
"#
    )
}
