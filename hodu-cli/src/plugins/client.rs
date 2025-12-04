//! JSON-RPC client for plugin communication
//!
//! This module provides the client-side JSON-RPC implementation for communicating
//! with plugin processes over stdio.

use hodu_plugin_sdk::rpc::{
    methods, BuildParams, CancelParams, InitializeParams, InitializeResult, ListTargetsResult, LoadModelParams,
    LoadModelResult, LoadTensorParams, LoadTensorResult, LogParams, Notification, ProgressParams, Request, RequestId,
    Response, RpcError, RunParams, RunResult, SaveModelParams, SaveTensorParams, TensorInput, JSONRPC_VERSION,
    PROTOCOL_VERSION,
};
use hodu_plugin_sdk::SDK_VERSION;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

/// Notification handler callback type
pub type NotificationHandler = Box<dyn Fn(&str, Option<&serde_json::Value>) + Send>;

/// Handle for cancelling requests from another thread (e.g., signal handler)
#[derive(Clone)]
pub struct CancellationHandle {
    stdin: Arc<Mutex<ChildStdin>>,
    current_request_id: Arc<AtomicI64>,
    next_id: Arc<AtomicI64>,
}

impl CancellationHandle {
    /// Cancel the current request if one is in progress
    pub fn cancel(&self) -> Result<(), ClientError> {
        let current_id = self.current_request_id.load(Ordering::SeqCst);
        if current_id == 0 {
            return Ok(()); // No request in progress
        }

        let params = CancelParams {
            id: RequestId::Number(current_id),
        };
        let request = Request::new(
            methods::CANCEL,
            Some(serde_json::to_value(params).map_err(ClientError::Serialize)?),
            RequestId::Number(self.next_id.fetch_add(1, Ordering::SeqCst)),
        );

        let mut stdin = self.stdin.lock().map_err(|_| ClientError::LockError)?;
        let json = serde_json::to_string(&request).map_err(ClientError::Serialize)?;
        writeln!(stdin, "{}", json).map_err(ClientError::Io)?;
        stdin.flush().map_err(ClientError::Io)?;
        Ok(())
    }
}

/// JSON-RPC client for communicating with a plugin process
pub struct PluginClient {
    stdin: Arc<Mutex<ChildStdin>>,
    stdout: BufReader<ChildStdout>,
    next_id: Arc<AtomicI64>,
    current_request_id: Arc<AtomicI64>,
    notification_handler: Option<NotificationHandler>,
}

impl PluginClient {
    /// Create a new client from a spawned child process
    pub fn new(child: &mut Child) -> Result<Self, ClientError> {
        let stdin = child.stdin.take().ok_or(ClientError::NoStdin)?;
        let stdout = child.stdout.take().ok_or(ClientError::NoStdout)?;

        Ok(Self {
            stdin: Arc::new(Mutex::new(stdin)),
            stdout: BufReader::new(stdout),
            next_id: Arc::new(AtomicI64::new(1)),
            current_request_id: Arc::new(AtomicI64::new(0)),
            notification_handler: None,
        })
    }

    /// Get a cancellation handle for use from another thread (e.g., Ctrl+C handler)
    pub fn cancellation_handle(&self) -> CancellationHandle {
        CancellationHandle {
            stdin: Arc::clone(&self.stdin),
            current_request_id: Arc::clone(&self.current_request_id),
            next_id: Arc::clone(&self.next_id),
        }
    }

    /// Use the default notification handler (prints to stderr)
    pub fn use_default_notification_handler(&mut self) {
        self.notification_handler = Some(Box::new(default_notification_handler));
    }

    /// Initialize the plugin and validate version compatibility
    pub fn initialize(&mut self) -> Result<InitializeResult, ClientError> {
        let params = InitializeParams {
            sdk_version: SDK_VERSION.to_string(),
            protocol_version: PROTOCOL_VERSION.to_string(),
        };

        let result: InitializeResult = self.call(methods::INITIALIZE, Some(params))?;

        // Validate protocol version compatibility
        // - For 0.x.y: major.minor must match (unstable API)
        // - For >= 1.0.0: major must match
        if !is_protocol_compatible(PROTOCOL_VERSION, &result.protocol_version) {
            return Err(ClientError::ProtocolMismatch {
                cli: PROTOCOL_VERSION.to_string(),
                plugin: result.protocol_version.clone(),
            });
        }

        // Warn if SDK versions differ (informational only)
        if result.sdk_version != SDK_VERSION {
            eprintln!(
                "Warning: Plugin '{}' SDK version ({}) differs from CLI ({})",
                result.name, result.sdk_version, SDK_VERSION
            );
        }

        Ok(result)
    }

    /// Shutdown the plugin gracefully
    pub fn shutdown(&mut self) -> Result<(), ClientError> {
        // Send shutdown request (plugin will exit)
        let request = Request::new(
            methods::SHUTDOWN,
            None,
            RequestId::Number(self.next_id.fetch_add(1, Ordering::SeqCst)),
        );
        let mut stdin = self.stdin.lock().map_err(|_| ClientError::LockError)?;
        let json = serde_json::to_string(&request).map_err(ClientError::Serialize)?;
        writeln!(stdin, "{}", json).map_err(ClientError::Io)?;
        stdin.flush().map_err(ClientError::Io)?;
        Ok(())
    }

    // ========================================================================
    // Format plugin methods
    // ========================================================================

    /// Load a model file using format plugin
    pub fn load_model(&mut self, path: &str) -> Result<LoadModelResult, ClientError> {
        let params = LoadModelParams { path: path.to_string() };
        self.call(methods::FORMAT_LOAD_MODEL, Some(params))
    }

    /// Save a model to file using format plugin
    pub fn save_model(&mut self, snapshot_path: &str, output_path: &str) -> Result<(), ClientError> {
        let params = SaveModelParams {
            snapshot_path: snapshot_path.to_string(),
            output_path: output_path.to_string(),
        };
        self.call::<_, serde_json::Value>(methods::FORMAT_SAVE_MODEL, Some(params))?;
        Ok(())
    }

    /// Load a tensor file using format plugin
    pub fn load_tensor(&mut self, path: &str) -> Result<LoadTensorResult, ClientError> {
        let params = LoadTensorParams { path: path.to_string() };
        self.call(methods::FORMAT_LOAD_TENSOR, Some(params))
    }

    /// Save a tensor to file using format plugin
    pub fn save_tensor(&mut self, tensor_path: &str, output_path: &str) -> Result<(), ClientError> {
        let params = SaveTensorParams {
            tensor_path: tensor_path.to_string(),
            output_path: output_path.to_string(),
        };
        self.call::<_, serde_json::Value>(methods::FORMAT_SAVE_TENSOR, Some(params))?;
        Ok(())
    }

    // ========================================================================
    // Backend plugin methods
    // ========================================================================

    /// Run model inference using backend plugin
    pub fn run(
        &mut self,
        library_path: &str,
        snapshot_path: &str,
        device: &str,
        inputs: Vec<TensorInput>,
    ) -> Result<RunResult, ClientError> {
        let params = RunParams {
            library_path: library_path.to_string(),
            snapshot_path: snapshot_path.to_string(),
            device: device.to_string(),
            inputs,
        };
        self.call(methods::BACKEND_RUN, Some(params))
    }

    /// Build (AOT compile) model using backend plugin
    pub fn build(
        &mut self,
        snapshot_path: &str,
        target: &str,
        device: &str,
        format: &str,
        output_path: &str,
    ) -> Result<(), ClientError> {
        let params = BuildParams {
            snapshot_path: snapshot_path.to_string(),
            target: target.to_string(),
            device: device.to_string(),
            format: format.to_string(),
            output_path: output_path.to_string(),
        };
        self.call::<_, serde_json::Value>(methods::BACKEND_BUILD, Some(params))?;
        Ok(())
    }

    /// List supported build targets
    pub fn list_targets(&mut self) -> Result<ListTargetsResult, ClientError> {
        self.call(methods::BACKEND_SUPPORTED_TARGETS, Some(serde_json::json!({})))
    }

    // ========================================================================
    // Internal
    // ========================================================================

    /// Send a request and wait for response, handling notifications
    fn call<P, R>(&mut self, method: &str, params: Option<P>) -> Result<R, ClientError>
    where
        P: serde::Serialize,
        R: serde::de::DeserializeOwned,
    {
        let id_num = self.next_id.fetch_add(1, Ordering::SeqCst);
        let id = RequestId::Number(id_num);

        // Track current request ID for cancellation
        self.current_request_id.store(id_num, Ordering::SeqCst);

        let params_value = params
            .map(|p| serde_json::to_value(p))
            .transpose()
            .map_err(ClientError::Serialize)?;

        let request = Request {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: method.to_string(),
            params: params_value,
            id: id.clone(),
        };

        // Send request
        {
            let mut stdin = self.stdin.lock().map_err(|_| ClientError::LockError)?;
            let json = serde_json::to_string(&request).map_err(ClientError::Serialize)?;
            writeln!(stdin, "{}", json).map_err(ClientError::Io)?;
            stdin.flush().map_err(ClientError::Io)?;
        }

        // Read response, handling notifications along the way
        loop {
            let mut line = String::new();
            self.stdout.read_line(&mut line).map_err(ClientError::Io)?;

            if line.is_empty() {
                return Err(ClientError::ConnectionClosed);
            }

            // Try to parse as JSON first
            let value: serde_json::Value =
                serde_json::from_str(&line).map_err(|e| ClientError::Parse(e.to_string()))?;

            // Check if it's a notification (no "id" field) or response (has "id" field)
            if value.get("id").is_none() {
                // It's a notification
                if let Ok(notification) = serde_json::from_value::<Notification>(value) {
                    self.handle_notification(&notification);
                }
                continue; // Keep reading for the actual response
            }

            // It's a response - clear current request ID
            self.current_request_id.store(0, Ordering::SeqCst);

            let response: Response = serde_json::from_str(&line).map_err(|e| ClientError::Parse(e.to_string()))?;

            // Verify ID matches
            if response.id != id {
                return Err(ClientError::IdMismatch);
            }

            // Check for error
            if let Some(error) = response.error {
                return Err(ClientError::Rpc(error));
            }

            // Parse result
            let result = response.result.ok_or(ClientError::NoResult)?;
            return serde_json::from_value(result).map_err(|e| ClientError::Parse(e.to_string()));
        }
    }

    /// Handle a notification from the plugin
    fn handle_notification(&self, notification: &Notification) {
        if let Some(handler) = &self.notification_handler {
            handler(&notification.method, notification.params.as_ref());
        }
    }
}

/// Default notification handler that prints to stderr
fn default_notification_handler(method: &str, params: Option<&serde_json::Value>) {
    match method {
        methods::NOTIFY_PROGRESS => {
            if let Some(params) = params {
                if let Ok(p) = serde_json::from_value::<ProgressParams>(params.clone()) {
                    if let Some(percent) = p.percent {
                        // Clear line and print progress
                        eprint!("\r\x1b[K[{:3}%] {}", percent, p.message);
                        if percent >= 100 {
                            eprintln!(); // newline after 100%
                        }
                    } else {
                        eprint!("\r\x1b[K[...] {}", p.message);
                    }
                    let _ = std::io::stderr().flush();
                }
            }
        },
        methods::NOTIFY_LOG => {
            if let Some(params) = params {
                if let Ok(p) = serde_json::from_value::<LogParams>(params.clone()) {
                    eprintln!("[{}] {}", p.level.to_uppercase(), p.message);
                }
            }
        },
        _ => {
            // Unknown notification, ignore
        },
    }
}

/// Client errors
#[derive(Debug)]
pub enum ClientError {
    NoStdin,
    NoStdout,
    Io(std::io::Error),
    Serialize(serde_json::Error),
    Parse(String),
    ConnectionClosed,
    IdMismatch,
    NoResult,
    Rpc(RpcError),
    ProtocolMismatch { cli: String, plugin: String },
    LockError,
}

/// Check if two protocol versions are compatible
/// - For 0.x.y: major.minor must match (unstable API)
/// - For >= 1.0.0: major must match
fn is_protocol_compatible(v1: &str, v2: &str) -> bool {
    let parts1: Vec<u32> = v1.split('.').filter_map(|s| s.parse().ok()).collect();
    let parts2: Vec<u32> = v2.split('.').filter_map(|s| s.parse().ok()).collect();

    let (major1, minor1) = (
        parts1.first().copied().unwrap_or(0),
        parts1.get(1).copied().unwrap_or(0),
    );
    let (major2, minor2) = (
        parts2.first().copied().unwrap_or(0),
        parts2.get(1).copied().unwrap_or(0),
    );

    if major1 == 0 || major2 == 0 {
        // Pre-1.0: major.minor must match
        major1 == major2 && minor1 == minor2
    } else {
        // Post-1.0: major must match
        major1 == major2
    }
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::NoStdin => write!(f, "Plugin process has no stdin"),
            ClientError::NoStdout => write!(f, "Plugin process has no stdout"),
            ClientError::Io(e) => write!(f, "IO error: {}", e),
            ClientError::Serialize(e) => write!(f, "Serialization error: {}", e),
            ClientError::Parse(e) => write!(f, "Parse error: {}", e),
            ClientError::ConnectionClosed => write!(f, "Plugin connection closed unexpectedly"),
            ClientError::IdMismatch => write!(f, "Response ID does not match request"),
            ClientError::NoResult => write!(f, "Response has no result"),
            ClientError::Rpc(e) => write!(f, "RPC error ({}): {}", e.code, e.message),
            ClientError::ProtocolMismatch { cli, plugin } => write!(
                f,
                "Protocol version mismatch: CLI ({}) incompatible with plugin ({})",
                cli, plugin
            ),
            ClientError::LockError => write!(f, "Failed to acquire lock"),
        }
    }
}

impl std::error::Error for ClientError {}
