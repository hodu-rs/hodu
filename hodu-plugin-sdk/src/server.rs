//! Plugin server framework for JSON-RPC communication over stdio
//!
//! This module provides the runtime for plugins to handle JSON-RPC requests.

use crate::rpc::{
    error_codes, methods, InitializeParams, InitializeResult, Notification, Request, RequestId, Response, RpcError,
    PROTOCOL_VERSION,
};
use crate::SDK_VERSION;
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};

// ============================================================================
// Notification helpers (can be called from handlers)
// ============================================================================

/// Send a progress notification to the CLI
///
/// # Arguments
/// * `percent` - Progress percentage (0-100), None for indeterminate
/// * `message` - Progress message
pub fn notify_progress(percent: Option<u8>, message: &str) {
    let notification = Notification::progress(percent, message);
    if let Ok(json) = serde_json::to_string(&notification) {
        let _ = writeln!(std::io::stdout(), "{}", json);
        let _ = std::io::stdout().flush();
    }
}

/// Send a log notification to the CLI
///
/// # Arguments
/// * `level` - Log level: "error", "warn", "info", "debug", "trace"
/// * `message` - Log message
pub fn notify_log(level: &str, message: &str) {
    let notification = Notification::log(level, message);
    if let Ok(json) = serde_json::to_string(&notification) {
        let _ = writeln!(std::io::stdout(), "{}", json);
        let _ = std::io::stdout().flush();
    }
}

/// Convenience functions for different log levels
pub fn log_error(message: &str) {
    notify_log("error", message);
}
pub fn log_warn(message: &str) {
    notify_log("warn", message);
}
pub fn log_info(message: &str) {
    notify_log("info", message);
}
pub fn log_debug(message: &str) {
    notify_log("debug", message);
}

// ============================================================================
// Plugin Server
// ============================================================================

/// Plugin server that handles JSON-RPC requests over stdio
pub struct PluginServer {
    name: String,
    version: String,
    capabilities: Vec<String>,
    model_extensions: Option<Vec<String>>,
    tensor_extensions: Option<Vec<String>>,
    devices: Option<Vec<String>>,
    handlers: HashMap<String, BoxedHandler>,
    initialized: bool,
}

/// Type-erased handler using boxed closure
type BoxedHandler = Box<dyn Fn(Option<serde_json::Value>) -> Result<serde_json::Value, RpcError> + Send>;

impl PluginServer {
    /// Create a new plugin server
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            capabilities: Vec::new(),
            model_extensions: None,
            tensor_extensions: None,
            devices: None,
            handlers: HashMap::new(),
            initialized: false,
        }
    }

    /// Set supported file extensions for model format plugins
    pub fn model_extensions(mut self, exts: Vec<&str>) -> Self {
        self.model_extensions = Some(exts.into_iter().map(String::from).collect());
        self
    }

    /// Set supported file extensions for tensor format plugins
    pub fn tensor_extensions(mut self, exts: Vec<&str>) -> Self {
        self.tensor_extensions = Some(exts.into_iter().map(String::from).collect());
        self
    }

    /// Set supported devices (for backend plugins)
    pub fn devices(mut self, devs: Vec<&str>) -> Self {
        self.devices = Some(devs.into_iter().map(String::from).collect());
        self
    }

    /// Register a method handler with params
    pub fn method<F, P, R>(mut self, name: &str, handler: F) -> Self
    where
        F: Fn(P) -> Result<R, RpcError> + Send + 'static,
        P: DeserializeOwned + 'static,
        R: Serialize + 'static,
    {
        // Auto-register capability based on method name
        if (name.starts_with("format.") || name.starts_with("backend."))
            && !self.capabilities.contains(&name.to_string())
        {
            self.capabilities.push(name.to_string());
        }

        // Create type-erased handler
        let boxed: BoxedHandler = Box::new(move |params| {
            let params: P = match params {
                Some(v) => serde_json::from_value(v).map_err(|e| RpcError::invalid_params(e.to_string()))?,
                None => return Err(RpcError::invalid_params("Missing params")),
            };
            let result = handler(params)?;
            serde_json::to_value(result).map_err(|e| RpcError::internal_error(e.to_string()))
        });

        self.handlers.insert(name.to_string(), boxed);
        self
    }

    /// Register a method handler with no params
    pub fn method_no_params<F, R>(mut self, name: &str, handler: F) -> Self
    where
        F: Fn() -> Result<R, RpcError> + Send + 'static,
        R: Serialize + 'static,
    {
        if (name.starts_with("format.") || name.starts_with("backend."))
            && !self.capabilities.contains(&name.to_string())
        {
            self.capabilities.push(name.to_string());
        }

        let boxed: BoxedHandler = Box::new(move |_params| {
            let result = handler()?;
            serde_json::to_value(result).map_err(|e| RpcError::internal_error(e.to_string()))
        });

        self.handlers.insert(name.to_string(), boxed);
        self
    }

    /// Run the server (blocking, reads from stdin, writes to stdout)
    pub fn run(mut self) -> Result<(), Box<dyn std::error::Error>> {
        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();
        let reader = BufReader::new(stdin.lock());

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            let response = self.handle_message(&line);
            if let Some(resp) = response {
                let json = serde_json::to_string(&resp)?;
                writeln!(stdout, "{}", json)?;
                stdout.flush()?;
            }
        }

        Ok(())
    }

    fn handle_message(&mut self, line: &str) -> Option<Response> {
        // Parse request
        let request: Request = match serde_json::from_str(line) {
            Ok(req) => req,
            Err(e) => {
                return Some(Response::error(
                    RequestId::Number(0),
                    RpcError::parse_error(e.to_string()),
                ));
            },
        };

        let id = request.id.clone();

        // Handle based on method
        let result = match request.method.as_str() {
            methods::INITIALIZE => self.handle_initialize(request.params),
            methods::SHUTDOWN => {
                // Graceful shutdown
                std::process::exit(0);
            },
            method => {
                if !self.initialized {
                    Err(RpcError::new(error_codes::INVALID_REQUEST, "Server not initialized"))
                } else if let Some(handler) = self.handlers.get(method) {
                    handler(request.params)
                } else {
                    Err(RpcError::method_not_found(method))
                }
            },
        };

        Some(match result {
            Ok(value) => Response::success(id, value),
            Err(error) => Response::error(id, error),
        })
    }

    fn handle_initialize(&mut self, params: Option<serde_json::Value>) -> Result<serde_json::Value, RpcError> {
        if self.initialized {
            return Err(RpcError::new(error_codes::INVALID_REQUEST, "Already initialized"));
        }

        let _params: InitializeParams = match params {
            Some(v) => serde_json::from_value(v).map_err(|e| RpcError::invalid_params(e.to_string()))?,
            None => return Err(RpcError::invalid_params("Missing params")),
        };

        self.initialized = true;

        let result = InitializeResult {
            name: self.name.clone(),
            version: self.version.clone(),
            protocol_version: PROTOCOL_VERSION.to_string(),
            sdk_version: SDK_VERSION.to_string(),
            capabilities: self.capabilities.clone(),
            model_extensions: self.model_extensions.clone(),
            tensor_extensions: self.tensor_extensions.clone(),
            devices: self.devices.clone(),
        };

        serde_json::to_value(result).map_err(|e| RpcError::internal_error(e.to_string()))
    }
}
