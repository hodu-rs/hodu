//! {{NAME}} - Backend plugin for Hodu

use hodu_plugin_sdk::{
    hdss,
    rpc::{RpcError, RunParams, RunResult},
    server::PluginServer,
    TensorData,
};
use std::collections::HashMap;

fn main() {
    let server = PluginServer::new("{{NAME}}", env!("CARGO_PKG_VERSION"))
        .devices(vec!["cpu"])
        .method("backend.run", handle_run);

    if let Err(e) = server.run() {
        eprintln!("Plugin error: {}", e);
        std::process::exit(1);
    }
}

fn handle_run(params: RunParams) -> Result<RunResult, RpcError> {
    let snapshot = hdss::load(&params.snapshot_path)
        .map_err(|e| RpcError::internal_error(format!("Failed to load snapshot: {}", e)))?;

    let mut inputs: HashMap<String, TensorData> = HashMap::new();
    for input in &params.inputs {
        let tensor = TensorData::load(&input.path)
            .map_err(|e| RpcError::internal_error(format!("Failed to load input '{}': {}", input.name, e)))?;
        inputs.insert(input.name.clone(), tensor);
    }

    // TODO: Implement your model execution logic here
    Err(RpcError::internal_error(format!(
        "Backend '{}' execution not implemented. Model has {} nodes, {} inputs provided.",
        params.device,
        snapshot.nodes.len(),
        inputs.len()
    )))
}
