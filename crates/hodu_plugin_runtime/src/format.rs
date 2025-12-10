//! Format plugin runtime
//!
//! This module provides the runtime for loading and executing format plugins.
//! Format plugins handle loading and saving models and tensors in various file formats.

use crate::client::{ClientError, PluginClient, DEFAULT_TIMEOUT};
use crate::registry::{PluginEntry, PluginRegistry, RegistryError};
use hodu_plugin::rpc::InitializeResult;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

/// Format plugin manager
///
/// Manages format plugin processes with keep-alive support for reuse.
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
    /// Create a new format plugin manager
    pub fn new() -> Result<Self, ManagerError> {
        let registry_path = PluginRegistry::default_path().map_err(ManagerError::Registry)?;
        let registry = PluginRegistry::load(&registry_path).map_err(ManagerError::Registry)?;
        let plugins_dir = PluginRegistry::plugins_dir().map_err(ManagerError::Registry)?;

        Ok(Self {
            processes: HashMap::new(),
            registry,
            plugins_dir,
            timeout: DEFAULT_TIMEOUT,
        })
    }

    /// Create a new plugin manager with a custom timeout
    pub fn with_timeout(timeout_secs: u64) -> Result<Self, ManagerError> {
        let mut manager = Self::new()?;
        manager.timeout = Duration::from_secs(timeout_secs);
        Ok(manager)
    }

    /// Set the timeout for plugin operations
    pub fn set_timeout(&mut self, timeout_secs: u64) {
        self.timeout = Duration::from_secs(timeout_secs);
    }

    /// Get or spawn a format plugin by name
    pub fn get_plugin(&mut self, name: &str) -> Result<&mut PluginClient, ManagerError> {
        // Check if already running
        if self.processes.contains_key(name) {
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
            .ok_or_else(|| ManagerError::NotFound(name.to_string()))?;

        // Check if plugin is enabled
        if !entry.enabled {
            return Err(ManagerError::Disabled(name.to_string()));
        }

        let entry = entry.clone();

        // Spawn plugin
        let managed = self.spawn_plugin(&entry)?;
        self.processes.insert(name.to_string(), managed);

        Ok(&mut self.processes.get_mut(name).expect("key exists after insert").client)
    }

    /// Get a format plugin by model extension
    pub fn get_for_model_extension(&mut self, ext: &str) -> Result<&mut PluginClient, ManagerError> {
        let entry = self
            .registry
            .find_model_format_by_extension(ext)
            .ok_or_else(|| ManagerError::NoFormatForExtension(ext.to_string()))?
            .clone();

        self.get_plugin(&entry.name)
    }

    /// Get a format plugin by tensor extension
    pub fn get_for_tensor_extension(&mut self, ext: &str) -> Result<&mut PluginClient, ManagerError> {
        let entry = self
            .registry
            .find_tensor_format_by_extension(ext)
            .ok_or_else(|| ManagerError::NoFormatForExtension(ext.to_string()))?
            .clone();

        self.get_plugin(&entry.name)
    }

    /// Spawn a plugin process
    fn spawn_plugin(&self, entry: &PluginEntry) -> Result<ManagedPlugin, ManagerError> {
        let binary_path = self.plugins_dir.join(&entry.name).join(&entry.binary);

        if !binary_path.exists() {
            return Err(ManagerError::BinaryNotFound(binary_path.to_string_lossy().to_string()));
        }

        // Spawn process
        let mut child = Command::new(&binary_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| ManagerError::Spawn(e.to_string()))?;

        // Create client
        let mut client = PluginClient::new(&mut child).map_err(ManagerError::Client)?;

        // Set timeout
        client.set_timeout(self.timeout);

        // Initialize
        let info = client.initialize().map_err(ManagerError::Client)?;

        Ok(ManagedPlugin { child, client, info })
    }

    /// Shutdown a specific plugin
    pub fn shutdown_plugin(&mut self, name: &str) -> Result<(), ManagerError> {
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
}

impl Drop for PluginManager {
    fn drop(&mut self) {
        self.shutdown_all();
    }
}

/// Format plugin manager errors
#[derive(Debug)]
pub enum ManagerError {
    Registry(RegistryError),
    NotFound(String),
    Disabled(String),
    NoFormatForExtension(String),
    BinaryNotFound(String),
    Spawn(String),
    Client(ClientError),
}

impl std::fmt::Display for ManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManagerError::Registry(e) => write!(f, "Registry error: {}", e),
            ManagerError::NotFound(name) => write!(f, "Plugin not found: {}", name),
            ManagerError::Disabled(name) => write!(
                f,
                "Plugin is disabled: {} (use `hodu plugin enable {}` to enable)",
                name, name
            ),
            ManagerError::NoFormatForExtension(ext) => {
                write!(f, "No format plugin found for extension: {}", ext)
            },
            ManagerError::BinaryNotFound(path) => {
                write!(f, "Plugin binary not found: {}", path)
            },
            ManagerError::Spawn(e) => write!(f, "Failed to spawn plugin: {}", e),
            ManagerError::Client(e) => write!(f, "Client error: {}", e),
        }
    }
}

impl std::error::Error for ManagerError {}
