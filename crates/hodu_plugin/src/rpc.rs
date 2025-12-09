//! JSON-RPC 2.0 protocol types for plugin communication
//!
//! This module defines the message types for CLI <-> Plugin communication over stdio.

use serde::{Deserialize, Serialize};

/// JSON-RPC version string
pub const JSONRPC_VERSION: &str = "2.0";

/// Plugin protocol version for compatibility checking
pub const PROTOCOL_VERSION: &str = "1.0.0";

// ============================================================================
// Core JSON-RPC Types
// ============================================================================

/// JSON-RPC request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
    pub id: RequestId,
}

/// JSON-RPC response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
    pub id: RequestId,
}

/// JSON-RPC notification message (no id, no response expected)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

/// Request ID (can be number or string)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum RequestId {
    Number(i64),
    String(String),
}

/// JSON-RPC error object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

// ============================================================================
// Error Codes (JSON-RPC standard + custom)
// ============================================================================

pub mod error_codes {
    // Standard JSON-RPC errors
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;

    // Custom plugin errors (-32000 to -32099)
    pub const PLUGIN_ERROR: i32 = -32000;
    pub const NOT_SUPPORTED: i32 = -32001;
    pub const FILE_NOT_FOUND: i32 = -32002;
    pub const INVALID_FORMAT: i32 = -32003;
    pub const DEVICE_NOT_AVAILABLE: i32 = -32004;
    pub const MODEL_ERROR: i32 = -32005;
    pub const TENSOR_ERROR: i32 = -32006;
    pub const REQUEST_CANCELLED: i32 = -32007;
}

// ============================================================================
// Method Names
// ============================================================================

pub mod methods {
    // Lifecycle
    pub const INITIALIZE: &str = "initialize";
    pub const SHUTDOWN: &str = "shutdown";

    // Format plugin methods
    pub const FORMAT_LOAD_MODEL: &str = "format.load_model";
    pub const FORMAT_SAVE_MODEL: &str = "format.save_model";
    pub const FORMAT_LOAD_TENSOR: &str = "format.load_tensor";
    pub const FORMAT_SAVE_TENSOR: &str = "format.save_tensor";

    // Backend plugin methods
    pub const BACKEND_RUN: &str = "backend.run";
    pub const BACKEND_BUILD: &str = "backend.build";
    pub const BACKEND_SUPPORTED_DEVICES: &str = "backend.supported_devices";
    pub const BACKEND_SUPPORTED_TARGETS: &str = "backend.supported_targets";

    // Notifications (plugin -> CLI)
    pub const NOTIFY_PROGRESS: &str = "$/progress";
    pub const NOTIFY_LOG: &str = "$/log";

    // Cancellation (CLI -> plugin)
    pub const CANCEL: &str = "$/cancel";
}

// ============================================================================
// Capability Identifiers
// ============================================================================

pub mod capabilities {
    pub const FORMAT_LOAD_MODEL: &str = "format.load_model";
    pub const FORMAT_SAVE_MODEL: &str = "format.save_model";
    pub const FORMAT_LOAD_TENSOR: &str = "format.load_tensor";
    pub const FORMAT_SAVE_TENSOR: &str = "format.save_tensor";
    pub const BACKEND_RUN: &str = "backend.run";
    pub const BACKEND_BUILD: &str = "backend.build";
    pub const BACKEND_SUPPORTED_TARGETS: &str = "backend.supported_targets";
}

// ============================================================================
// Request/Response Params
// ============================================================================

/// Initialize request params
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeParams {
    pub plugin_version: String,
    pub protocol_version: String,
}

/// Initialize response result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResult {
    pub name: String,
    pub version: String,
    pub protocol_version: String,
    pub plugin_version: String,
    pub capabilities: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_extensions: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_extensions: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub devices: Option<Vec<String>>,
}

/// Load model request params
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelParams {
    pub path: String,
}

/// Load model response result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelResult {
    pub snapshot_path: String,
}

/// Save model request params
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveModelParams {
    pub snapshot_path: String,
    pub output_path: String,
}

/// Load tensor request params
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTensorParams {
    pub path: String,
}

/// Load tensor response result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTensorResult {
    pub tensor_path: String,
}

/// Save tensor request params
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveTensorParams {
    pub tensor_path: String,
    pub output_path: String,
}

/// Backend run request params
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunParams {
    /// Path to compiled library (.dylib, .so, .dll)
    pub library_path: String,
    /// Path to snapshot (needed for input/output metadata)
    pub snapshot_path: String,
    pub device: String,
    pub inputs: Vec<TensorInput>,
}

/// Input tensor reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInput {
    pub name: String,
    pub path: String,
}

/// Backend run response result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub outputs: Vec<TensorOutput>,
}

/// Output tensor reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorOutput {
    pub name: String,
    pub path: String,
}

/// Backend build request params
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildParams {
    pub snapshot_path: String,
    pub target: String,
    pub device: String,
    pub format: String,
    pub output_path: String,
}

/// Backend list targets response result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListTargetsResult {
    /// List of supported target triples
    pub targets: Vec<String>,
    /// Human-readable formatted list of targets (for display)
    pub formatted: String,
}

/// Progress notification params (plugin -> CLI)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressParams {
    /// Progress percentage (0-100), None for indeterminate
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub percent: Option<u8>,
    /// Progress message
    pub message: String,
}

/// Log notification params (plugin -> CLI)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogParams {
    /// Log level: "error", "warn", "info", "debug", "trace"
    pub level: String,
    /// Log message
    pub message: String,
}

/// Cancel request params (CLI -> plugin)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelParams {
    /// Request ID to cancel
    pub id: RequestId,
}

// ============================================================================
// Helper Implementations
// ============================================================================

impl Request {
    pub fn new(method: impl Into<String>, params: Option<serde_json::Value>, id: RequestId) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: method.into(),
            params,
            id,
        }
    }
}

impl Response {
    pub fn success(id: RequestId, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            result: Some(result),
            error: None,
            id,
        }
    }

    pub fn error(id: RequestId, error: RpcError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            result: None,
            error: Some(error),
            id,
        }
    }
}

impl Notification {
    pub fn new(method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: method.into(),
            params,
        }
    }

    pub fn progress(percent: Option<u8>, message: impl Into<String>) -> Self {
        let params = ProgressParams {
            percent,
            message: message.into(),
        };
        Self::new(methods::NOTIFY_PROGRESS, Some(serde_json::to_value(params).unwrap()))
    }

    pub fn log(level: impl Into<String>, message: impl Into<String>) -> Self {
        let params = LogParams {
            level: level.into(),
            message: message.into(),
        };
        Self::new(methods::NOTIFY_LOG, Some(serde_json::to_value(params).unwrap()))
    }
}

impl RpcError {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    pub fn with_data(code: i32, message: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            code,
            message: message.into(),
            data: Some(data),
        }
    }

    pub fn parse_error(msg: impl Into<String>) -> Self {
        Self::new(error_codes::PARSE_ERROR, msg)
    }

    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self::new(error_codes::INVALID_REQUEST, msg)
    }

    pub fn method_not_found(method: &str) -> Self {
        Self::new(error_codes::METHOD_NOT_FOUND, format!("Method not found: {}", method))
    }

    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self::new(error_codes::INVALID_PARAMS, msg)
    }

    pub fn internal_error(msg: impl Into<String>) -> Self {
        Self::new(error_codes::INTERNAL_ERROR, msg)
    }

    pub fn not_supported(feature: &str) -> Self {
        Self::new(error_codes::NOT_SUPPORTED, format!("Not supported: {}", feature))
    }

    pub fn file_not_found(path: &str) -> Self {
        Self::new(error_codes::FILE_NOT_FOUND, format!("File not found: {}", path))
    }

    pub fn cancelled() -> Self {
        Self::new(error_codes::REQUEST_CANCELLED, "Request cancelled")
    }
}

impl From<i64> for RequestId {
    fn from(n: i64) -> Self {
        RequestId::Number(n)
    }
}

impl From<String> for RequestId {
    fn from(s: String) -> Self {
        RequestId::String(s)
    }
}

impl From<&str> for RequestId {
    fn from(s: &str) -> Self {
        RequestId::String(s.to_string())
    }
}
