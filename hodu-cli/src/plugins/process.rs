//! Plugin process management
//!
//! This module handles spawning, managing, and keeping alive plugin processes.
//! Plugins are standalone executables that communicate via JSON-RPC over stdio.

use super::client::{CancellationHandle, ClientError, PluginClient, DEFAULT_TIMEOUT};
use super::registry::{PluginEntry, PluginRegistry, RegistryError};
use hodu_plugin_sdk::rpc::InitializeResult;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

/// Plugin process manager
///
/// Manages plugin processes with keep-alive support for reuse.
pub struct PluginManager {
    /// Running plugin processes (name -> process)
    processes: HashMap<String, ManagedPlugin>,
    /// Plugin registry
    registry: PluginRegistry,
    /// Plugins directory
    plugins_dir: PathBuf,
    /// Timeout for plugin operations
    timeout: Duration,
}

/// A managed plugin process
struct ManagedPlugin {
    child: Child,
    client: PluginClient,
    info: InitializeResult,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Result<Self, ProcessError> {
        let registry_path = PluginRegistry::default_path().map_err(ProcessError::Registry)?;
        let registry = PluginRegistry::load(&registry_path).map_err(ProcessError::Registry)?;
        let plugins_dir = PluginRegistry::plugins_dir().map_err(ProcessError::Registry)?;

        Ok(Self {
            processes: HashMap::new(),
            registry,
            plugins_dir,
            timeout: DEFAULT_TIMEOUT,
        })
    }

    /// Create a new plugin manager with a custom timeout
    pub fn with_timeout(timeout_secs: u64) -> Result<Self, ProcessError> {
        let mut manager = Self::new()?;
        manager.timeout = Duration::from_secs(timeout_secs);
        Ok(manager)
    }

    /// Set the timeout for plugin operations
    pub fn set_timeout(&mut self, timeout_secs: u64) {
        self.timeout = Duration::from_secs(timeout_secs);
    }

    /// Get or spawn a plugin by name
    pub fn get_plugin(&mut self, name: &str) -> Result<&mut PluginClient, ProcessError> {
        // Check if already running
        if self.processes.contains_key(name) {
            // SAFETY: We just checked that the key exists
            return Ok(&mut self
                .processes
                .get_mut(name)
                .expect("key exists after contains_key check")
                .client);
        }

        // Find plugin in registry
        let entry = self
            .registry
            .find(name)
            .ok_or_else(|| ProcessError::NotFound(name.to_string()))?;

        // Check if plugin is enabled
        if !entry.enabled {
            return Err(ProcessError::Disabled(name.to_string()));
        }

        let entry = entry.clone();

        // Spawn plugin
        let managed = self.spawn_plugin(&entry)?;
        self.processes.insert(name.to_string(), managed);

        // SAFETY: We just inserted the key
        Ok(&mut self.processes.get_mut(name).expect("key exists after insert").client)
    }

    /// Get a format plugin by extension (tries model format first, then tensor format)
    pub fn get_format_for_extension(&mut self, ext: &str) -> Result<&mut PluginClient, ProcessError> {
        let entry = self
            .registry
            .find_model_format_by_extension(ext)
            .or_else(|| self.registry.find_tensor_format_by_extension(ext))
            .ok_or_else(|| ProcessError::NoFormatForExtension(ext.to_string()))?
            .clone();

        self.get_plugin(&entry.name)
    }

    /// Spawn a plugin process
    fn spawn_plugin(&self, entry: &PluginEntry) -> Result<ManagedPlugin, ProcessError> {
        let binary_path = self.plugins_dir.join(&entry.name).join(&entry.binary);

        if !binary_path.exists() {
            return Err(ProcessError::BinaryNotFound(binary_path.to_string_lossy().to_string()));
        }

        // Spawn process
        let mut child = Command::new(&binary_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| ProcessError::Spawn(e.to_string()))?;

        // Create client
        let mut client = PluginClient::new(&mut child).map_err(ProcessError::Client)?;

        // Set timeout
        client.set_timeout(self.timeout);

        // Enable notification handling (progress, logs)
        client.use_default_notification_handler();

        // Initialize
        let info = client.initialize().map_err(ProcessError::Client)?;

        Ok(ManagedPlugin { child, client, info })
    }

    /// Shutdown a specific plugin
    pub fn shutdown_plugin(&mut self, name: &str) -> Result<(), ProcessError> {
        if let Some(mut managed) = self.processes.remove(name) {
            let _ = managed.client.shutdown();
            let _ = managed.child.wait();
        }
        Ok(())
    }

    /// Shutdown all plugins
    pub fn shutdown_all(&mut self) {
        let names: Vec<String> = self.processes.keys().cloned().collect();
        for name in names {
            let _ = self.shutdown_plugin(&name);
        }
    }

    /// Get plugin info if running
    pub fn get_info(&self, name: &str) -> Option<&InitializeResult> {
        self.processes.get(name).map(|p| &p.info)
    }

    /// Get cancellation handle for a plugin (for Ctrl+C handling)
    pub fn get_cancellation_handle(&self, name: &str) -> Option<CancellationHandle> {
        self.processes.get(name).map(|p| p.client.cancellation_handle())
    }
}

impl Drop for PluginManager {
    fn drop(&mut self) {
        self.shutdown_all();
    }
}

/// Process management errors
#[derive(Debug)]
pub enum ProcessError {
    Registry(RegistryError),
    NotFound(String),
    Disabled(String),
    NoFormatForExtension(String),
    BinaryNotFound(String),
    Spawn(String),
    Client(ClientError),
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessError::Registry(e) => write!(f, "Registry error: {}", e),
            ProcessError::NotFound(name) => write!(f, "Plugin not found: {}", name),
            ProcessError::Disabled(name) => write!(
                f,
                "Plugin is disabled: {} (use `hodu plugin enable {}` to enable)",
                name, name
            ),
            ProcessError::NoFormatForExtension(ext) => {
                write!(f, "No format plugin found for extension: {}", ext)
            },
            ProcessError::BinaryNotFound(path) => {
                write!(f, "Plugin binary not found: {}", path)
            },
            ProcessError::Spawn(e) => write!(f, "Failed to spawn plugin: {}", e),
            ProcessError::Client(e) => write!(f, "Client error: {}", e),
        }
    }
}

impl std::error::Error for ProcessError {}
