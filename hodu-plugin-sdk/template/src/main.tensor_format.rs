//! {{NAME}} - Tensor format plugin for Hodu

use hodu_plugin_sdk::{
    rpc::{LoadTensorParams, LoadTensorResult, RpcError},
    server::PluginServer,
    TensorData,
};
use std::path::Path;

fn main() {
    let server = PluginServer::new("{{NAME}}", env!("CARGO_PKG_VERSION"))
        .tensor_extensions(vec!["ext"])
        .method("format.load_tensor", handle_load_tensor);

    if let Err(e) = server.run() {
        eprintln!("Plugin error: {}", e);
        std::process::exit(1);
    }
}

fn handle_load_tensor(params: LoadTensorParams) -> Result<LoadTensorResult, RpcError> {
    let path = Path::new(&params.path);

    if !path.exists() {
        return Err(RpcError::invalid_params(format!("File not found: {}", params.path)));
    }

    // TODO: Implement your tensor parsing logic here
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    Err(RpcError::internal_error(format!(
        "Tensor format '{}' parsing not implemented. File: {} ({} bytes)",
        path.extension().and_then(|e| e.to_str()).unwrap_or("unknown"),
        params.path,
        file_size
    )))
}
