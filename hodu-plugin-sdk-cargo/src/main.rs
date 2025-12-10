//! cargo-hodu-plugin-sdk - Cargo subcommand for creating Hodu plugins
//!
//! Usage:
//!   cargo hodu-plugin-sdk init <name> --type <type>

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Plugin protocol version
const PLUGIN_VERSION: &str = "0.3.0";

#[derive(Parser)]
#[command(name = "cargo")]
#[command(bin_name = "cargo")]
struct Cli {
    #[command(subcommand)]
    command: CargoCommand,
}

#[derive(Subcommand)]
enum CargoCommand {
    /// Hodu plugin SDK tools
    HoduPluginSdk {
        #[command(subcommand)]
        command: Command,
    },
}

#[derive(Subcommand)]
enum Command {
    /// Initialize a new plugin project
    Init {
        /// Plugin name
        name: String,

        /// Plugin type: backend, model_format, or tensor_format
        #[arg(short = 't', long = "type")]
        plugin_type: String,

        /// Output directory (default: current directory)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();

    let CargoCommand::HoduPluginSdk { command } = cli.command;

    let result = match command {
        Command::Init {
            name,
            plugin_type,
            output,
        } => create_plugin(&name, &plugin_type, output),
    };

    if let Err(e) = result {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

fn create_plugin(name: &str, plugin_type: &str, output: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let plugin_type = plugin_type.to_lowercase();
    let valid_types = ["backend", "model_format", "tensor_format"];
    if !valid_types.contains(&plugin_type.as_str()) {
        return Err(format!(
            "Invalid plugin type: '{}'. Use 'backend', 'model_format', or 'tensor_format'.",
            plugin_type
        )
        .into());
    }

    let output_dir = match output {
        Some(dir) => dir,
        None => std::env::current_dir()?,
    };
    let project_dir = output_dir.join(name);

    if project_dir.exists() {
        return Err(format!("Directory already exists: {}", project_dir.display()).into());
    }

    println!("Creating {} plugin: {}", plugin_type, name);

    std::fs::create_dir_all(&project_dir)?;
    std::fs::create_dir_all(project_dir.join("src"))?;

    // Cargo.toml
    let cargo_toml = cargo_toml_template(name);
    std::fs::write(project_dir.join("Cargo.toml"), cargo_toml)?;

    // manifest.json
    let manifest = match plugin_type.as_str() {
        "backend" => manifest_json_backend_template(name),
        "model_format" => manifest_json_model_format_template(name),
        "tensor_format" => manifest_json_tensor_format_template(name),
        _ => unreachable!(),
    };
    std::fs::write(project_dir.join("manifest.json"), manifest)?;

    // main.rs
    let main_rs = match plugin_type.as_str() {
        "backend" => main_rs_backend_template(name),
        "model_format" => main_rs_model_format_template(name),
        "tensor_format" => main_rs_tensor_format_template(name),
        _ => unreachable!(),
    };
    std::fs::write(project_dir.join("src").join("main.rs"), main_rs)?;

    println!("Created plugin project at: {}", project_dir.display());
    println!();
    println!("Next steps:");
    println!("  1. cd {}", name);
    println!("  2. Edit manifest.json with your plugin details");
    println!("  3. Implement the plugin in src/main.rs");
    println!("  4. Install with: hodu plugin install --path .");

    Ok(())
}

// =============================================================================
// Templates
// =============================================================================

fn manifest_json_backend_template(name: &str) -> String {
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

fn manifest_json_model_format_template(name: &str) -> String {
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

fn manifest_json_tensor_format_template(name: &str) -> String {
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

fn cargo_toml_template(name: &str) -> String {
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
# hodu_cpu_kernels = "0.3"
"#
    )
}

fn main_rs_backend_template(name: &str) -> String {
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

fn main_rs_model_format_template(name: &str) -> String {
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

fn main_rs_tensor_format_template(name: &str) -> String {
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
