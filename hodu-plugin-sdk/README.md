# hodu-plugin-sdk

[![Crates.io](https://img.shields.io/crates/v/hodu-plugin-sdk.svg)](https://crates.io/crates/hodu-plugin-sdk)
[![Doc.rs](https://docs.rs/hodu-plugin-sdk/badge.svg)](https://docs.rs/hodu-plugin-sdk)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/daminstudio/hodu#license)

SDK for building Hodu ML toolkit plugins.

## Overview

Plugins communicate with the Hodu CLI via JSON-RPC 2.0 over stdio. Each plugin runs as a separate process, providing isolation and language-agnostic extensibility.

## Plugin Types

| Type | Description | Capabilities |
|------|-------------|--------------|
| `backend` | Execute/compile models on devices | `backend.run`, `backend.build` |
| `model_format` | Load/save model files | `format.load_model`, `format.save_model` |
| `tensor_format` | Load/save tensor files | `format.load_tensor`, `format.save_tensor` |

## Quick Start

### Create a Plugin Project

First, install the SDK (includes the scaffolding tool):

```bash
cargo install hodu-plugin-sdk
```

Then create a new plugin project:

```bash
# Backend plugin (e.g., CUDA, Metal)
cargo hodu-plugin-sdk init my-backend -t backend

# Model format plugin (e.g., ONNX loader)
cargo hodu-plugin-sdk init my-format -t model_format

# Tensor format plugin (e.g., NPY loader)
cargo hodu-plugin-sdk init my-tensor -t tensor_format
```

### Backend Plugin Example

```rust
use hodu_plugin_sdk::server::PluginServer;
use hodu_plugin_sdk::rpc::{RunParams, RunResult, BuildParams, TensorOutput, RpcError};

fn handle_run(params: RunParams) -> Result<RunResult, RpcError> {
    // Load model from params.snapshot_path
    // Load inputs from params.inputs[].path
    // Run inference on params.device (e.g., "cpu", "cuda::0")
    // Save outputs to temp files

    Ok(RunResult {
        outputs: vec![
            TensorOutput {
                name: "output".to_string(),
                path: "/tmp/output.hdt".to_string(),
            }
        ],
    })
}

fn handle_build(params: BuildParams) -> Result<serde_json::Value, RpcError> {
    // Compile model from params.snapshot_path
    // Target: params.target, Device: params.device
    // Format: params.format (dylib, staticlib, metallib)
    // Save to params.output_path

    Ok(serde_json::json!({}))
}

fn main() {
    PluginServer::new("my-backend", env!("CARGO_PKG_VERSION"))
        .devices(vec!["cpu"])
        .method("backend.run", handle_run)
        .method("backend.build", handle_build)
        .run()
        .unwrap();
}
```

### Model Format Plugin Example

```rust
use hodu_plugin_sdk::server::PluginServer;
use hodu_plugin_sdk::rpc::{LoadModelParams, LoadModelResult, SaveModelParams, RpcError};

fn handle_load_model(params: LoadModelParams) -> Result<LoadModelResult, RpcError> {
    // Parse model file from params.path
    // Convert to Hodu snapshot format
    // Save snapshot to temp file

    Ok(LoadModelResult {
        snapshot_path: "/tmp/model.hdss".to_string(),
    })
}

fn handle_save_model(params: SaveModelParams) -> Result<serde_json::Value, RpcError> {
    // Load snapshot from params.snapshot_path
    // Convert to target format
    // Save to params.output_path

    Ok(serde_json::json!({}))
}

fn main() {
    PluginServer::new("my-format", env!("CARGO_PKG_VERSION"))
        .model_extensions(vec!["myformat", "mf"])
        .method("format.load_model", handle_load_model)
        .method("format.save_model", handle_save_model)
        .run()
        .unwrap();
}
```

### Tensor Format Plugin Example

```rust
use hodu_plugin_sdk::server::PluginServer;
use hodu_plugin_sdk::rpc::{LoadTensorParams, LoadTensorResult, SaveTensorParams, RpcError};

fn handle_load_tensor(params: LoadTensorParams) -> Result<LoadTensorResult, RpcError> {
    // Parse tensor file from params.path
    // Convert to Hodu tensor format (.hdt)
    // Save to temp file

    Ok(LoadTensorResult {
        tensor_path: "/tmp/tensor.hdt".to_string(),
    })
}

fn handle_save_tensor(params: SaveTensorParams) -> Result<serde_json::Value, RpcError> {
    // Load tensor from params.tensor_path
    // Convert to target format
    // Save to params.output_path

    Ok(serde_json::json!({}))
}

fn main() {
    PluginServer::new("my-tensor", env!("CARGO_PKG_VERSION"))
        .tensor_extensions(vec!["npy", "npz"])
        .method("format.load_tensor", handle_load_tensor)
        .method("format.save_tensor", handle_save_tensor)
        .run()
        .unwrap();
}
```

## API Reference

### PluginServer

Main server struct for handling JSON-RPC requests.

```rust
PluginServer::new(name: &str, version: &str) -> Self
    .model_extensions(exts: Vec<&str>) -> Self   // File extensions (model format plugins)
    .tensor_extensions(exts: Vec<&str>) -> Self  // File extensions (tensor format plugins)
    .devices(devs: Vec<&str>) -> Self        // Supported devices (backend plugins)
    .method(name: &str, handler: F) -> Self  // Register method handler
    .run() -> Result<(), Box<dyn Error>>     // Start server loop
```

### RpcError

Error type for method handlers.

```rust
RpcError::new(code: i32, message: &str) -> Self
RpcError::invalid_params(msg: &str) -> Self
RpcError::internal_error(msg: &str) -> Self
RpcError::not_supported(feature: &str) -> Self
RpcError::file_not_found(path: &str) -> Self
```

### TensorData

Tensor data structure for I/O operations.

```rust
pub struct TensorData {
    pub shape: Vec<usize>,
    pub dtype: PluginDType,
    pub data: Vec<u8>,
}
```

### PluginDType

Supported data types.

```rust
pub enum PluginDType {
    Bool,
    F8E4M3, F8E5M2,
    BF16, F16, F32, F64,
    U8, U16, U32, U64,
    I8, I16, I32, I64,
}
```

## JSON-RPC Protocol

### Lifecycle

```
CLI                          Plugin
 |                              |
 |-- initialize --------------->|
 |<-- {name, version, caps} ----|
 |                              |
 |-- method.call -------------->|
 |<-- $/progress (optional) ----|  (notifications)
 |<-- $/log (optional) ---------|
 |<-- result/error -------------|
 |                              |
 |-- $/cancel (optional) ------>|  (abort request)
 |                              |
 |-- shutdown ----------------->|
 |                         [exit]
```

### Method Names

| Method | Description |
|--------|-------------|
| `initialize` | Initialize plugin, returns capabilities |
| `shutdown` | Graceful shutdown |
| `format.load_model` | Load model file to snapshot |
| `format.save_model` | Save snapshot to model file |
| `format.load_tensor` | Load tensor file to .hdt |
| `format.save_tensor` | Save .hdt to tensor file |
| `backend.run` | Run model inference |
| `backend.build` | AOT compile model |
| `$/progress` | Progress notification (plugin → CLI) |
| `$/log` | Log notification (plugin → CLI) |
| `$/cancel` | Cancel request (CLI → plugin) |

### Error Codes

| Code | Name | Description |
|------|------|-------------|
| -32700 | Parse Error | Invalid JSON |
| -32600 | Invalid Request | Invalid JSON-RPC |
| -32601 | Method Not Found | Unknown method |
| -32602 | Invalid Params | Invalid parameters |
| -32603 | Internal Error | Internal plugin error |
| -32000 | Plugin Error | Generic plugin error |
| -32001 | Not Supported | Feature not supported |
| -32002 | File Not Found | File does not exist |
| -32003 | Invalid Format | Invalid file format |
| -32004 | Device Not Available | Device unavailable |
| -32007 | Request Cancelled | Request was cancelled |

## Notifications

Plugins can send notifications to the CLI during long-running operations. Notifications are one-way messages (no response expected).

### Progress Notification

Report progress during operations:

```rust
use hodu_plugin_sdk::{notify_progress, log_info};

fn handle_run(params: RunParams) -> Result<RunResult, RpcError> {
    notify_progress(Some(0), "Loading model...");
    // ... load model ...

    notify_progress(Some(50), "Running inference...");
    // ... run inference ...

    notify_progress(Some(100), "Done");
    Ok(result)
}
```

### Log Notification

Send log messages to the CLI:

```rust
use hodu_plugin_sdk::{log_info, log_warn, log_error, log_debug};

log_info("Processing input tensor");
log_warn("Large tensor detected, may be slow");
log_error("Failed to allocate memory");
log_debug("Tensor shape: [1, 3, 224, 224]");
```

### Notification Methods

| Method | Description | Params |
|--------|-------------|--------|
| `$/progress` | Progress update | `{percent?: 0-100, message: string}` |
| `$/log` | Log message | `{level: string, message: string}` |

## Cancellation

The CLI can send a cancel request (`$/cancel`) to abort a running operation. Plugins should periodically check for cancellation during long operations.

When cancelled, plugins should return `RpcError::cancelled()`.

## Version Compatibility

The CLI validates `protocol_version` returned by plugins during initialization.

### Protocol Version Rules

| Version Range | Compatibility Rule | Example |
|---------------|-------------------|---------|
| `0.x.y` | major.minor must match | `0.1.0` ↔ `0.1.5` OK, `0.1.0` ↔ `0.2.0` Error |
| `>= 1.0.0` | major must match | `1.0.0` ↔ `1.9.0` OK, `1.0.0` ↔ `2.0.0` Error |

### Plugin Version

Plugin version differences only emit a warning and do not block execution. This allows plugins built with slightly different SDK versions to still work together.

## Device Naming Convention

Devices use lowercase names with `::` separator for device index:

| Device | Format |
|--------|--------|
| CPU | `cpu` |
| CUDA | `cuda`, `cuda::0`, `cuda::1` |
| Metal | `metal` |
| ROCm | `rocm`, `rocm::0` |
| Vulkan | `vulkan` |
| WebGPU | `webgpu` |

## License

BSD-3-Clause
