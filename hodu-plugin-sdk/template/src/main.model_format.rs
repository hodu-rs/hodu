//! {{NAME}} - Model format plugin for Hodu

use hodu_plugin_sdk::{
    rpc::{LoadModelParams, LoadModelResult, RpcError},
    server::PluginServer,
};
use std::path::Path;

fn main() {
    let server = PluginServer::new("{{NAME}}", env!("CARGO_PKG_VERSION"))
        .model_extensions(vec!["ext"])
        .method("format.load_model", handle_load_model);

    if let Err(e) = server.run() {
        eprintln!("Plugin error: {}", e);
        std::process::exit(1);
    }
}

fn handle_load_model(params: LoadModelParams) -> Result<LoadModelResult, RpcError> {
    let path = Path::new(&params.path);

    if !path.exists() {
        return Err(RpcError::invalid_params(format!("File not found: {}", params.path)));
    }

    // TODO: Implement your model parsing logic here
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    Err(RpcError::internal_error(format!(
        "Model format '{}' parsing not implemented. File: {} ({} bytes)",
        path.extension().and_then(|e| e.to_str()).unwrap_or("unknown"),
        params.path,
        file_size
    )))
}
