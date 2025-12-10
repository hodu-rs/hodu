//! High-level runtime API for model loading and execution
//!
//! This module provides a simple, unified API for loading and running models
//! using the plugin system.
//!
//! # Example
//!
//! ```ignore
//! use hodu::prelude::*;
//! use hodu::plugin::Runtime;
//!
//! let mut runtime = Runtime::new()?;
//! let model = runtime.load("model.onnx")?;
//!
//! let input = Tensor::randn(&[1, 3, 224, 224], 0f32, 1.)?;
//! let outputs = runtime.run(&model, &[("input", &input)], "cpu", "")?;
//! ```

use crate::backend;
use crate::client::ClientError;
use crate::format;
use hodu_core::format::hdt;
use hodu_core::tensor::Tensor;
use hodu_core::types::DType;
use hodu_plugin::rpc::TensorInput;
use hodu_plugin::tensor::PluginDType;
use std::path::{Path, PathBuf};

/// High-level runtime for model loading and execution
pub struct Runtime {
    format_manager: format::PluginManager,
    backend_manager: backend::PluginManager,
    cache_dir: PathBuf,
}

impl Runtime {
    /// Create a new runtime with default settings
    pub fn new() -> Result<Self, RuntimeError> {
        let format_manager = format::PluginManager::new().map_err(RuntimeError::Format)?;
        let backend_manager = backend::PluginManager::new().map_err(RuntimeError::Backend)?;

        let cache_dir = dirs::home_dir()
            .ok_or_else(|| RuntimeError::Other("Could not find home directory".to_string()))?
            .join(".hodu")
            .join("cache");

        Ok(Self {
            format_manager,
            backend_manager,
            cache_dir,
        })
    }

    /// Create a new runtime with custom timeout (in seconds)
    pub fn with_timeout(timeout_secs: u64) -> Result<Self, RuntimeError> {
        let format_manager = format::PluginManager::with_timeout(timeout_secs).map_err(RuntimeError::Format)?;
        let backend_manager = backend::PluginManager::with_timeout(timeout_secs).map_err(RuntimeError::Backend)?;

        let cache_dir = dirs::home_dir()
            .ok_or_else(|| RuntimeError::Other("Could not find home directory".to_string()))?
            .join(".hodu")
            .join("cache");

        Ok(Self {
            format_manager,
            backend_manager,
            cache_dir,
        })
    }

    /// Set timeout for plugin operations
    pub fn set_timeout(&mut self, timeout_secs: u64) {
        self.format_manager.set_timeout(timeout_secs);
        self.backend_manager.set_timeout(timeout_secs);
    }

    /// Load a model from file
    ///
    /// The file format is determined by the extension. For `.hdss` files,
    /// no format plugin is needed. For other formats (e.g., `.onnx`),
    /// the appropriate format plugin must be installed.
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<Model, RuntimeError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(RuntimeError::FileNotFound(path.to_string_lossy().to_string()));
        }

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

        let snapshot_path = if ext == "hdss" {
            // Native format - no conversion needed
            path.to_path_buf()
        } else {
            // Use format plugin to convert
            let client = self
                .format_manager
                .get_for_model_extension(&ext)
                .map_err(RuntimeError::Format)?;

            let result = client
                .load_model(path.to_string_lossy().as_ref())
                .map_err(RuntimeError::Client)?;

            PathBuf::from(result.snapshot_path)
        };

        Ok(Model {
            snapshot_path,
            cache_dir: self.cache_dir.clone(),
            is_temp: ext != "hdss",
        })
    }

    /// Run inference on a model
    ///
    /// # Arguments
    ///
    /// * `model` - The loaded model to run
    /// * `inputs` - Named input tensors
    /// * `device` - Device to run on (e.g., "cpu", "cuda::0", "metal")
    /// * `backend` - Backend plugin name (e.g., "aot-cpu"). If empty or not found, auto-selects based on device.
    ///
    /// # Returns
    ///
    /// Vector of output tensors with their names
    pub fn run(
        &mut self,
        model: &Model,
        inputs: &[(&str, &Tensor)],
        device: &str,
        backend: &str,
    ) -> Result<Vec<(String, Tensor)>, RuntimeError> {
        // Get backend - try explicit name first, fall back to device-based selection
        let client = if !backend.is_empty() {
            match self.backend_manager.get_plugin(backend) {
                Ok(c) => c,
                Err(_) => self
                    .backend_manager
                    .get_for_device(device)
                    .map_err(RuntimeError::Backend)?,
            }
        } else {
            self.backend_manager
                .get_for_device(device)
                .map_err(RuntimeError::Backend)?
        };

        // Prepare inputs - save to temp files
        let temp_dir = std::env::temp_dir().join("hodu_runtime");
        std::fs::create_dir_all(&temp_dir).map_err(|e| RuntimeError::Io(e.to_string()))?;

        let mut tensor_inputs = Vec::new();
        for (name, tensor) in inputs {
            let input_path = temp_dir.join(format!("{}.hdt", name));
            hdt::save(tensor, &input_path)
                .map_err(|e| RuntimeError::Other(format!("Failed to save input tensor: {}", e)))?;

            tensor_inputs.push(TensorInput {
                name: name.to_string(),
                path: input_path.to_string_lossy().to_string(),
            });
        }

        // Run inference
        let result = client
            .run(
                "", // lib_path - empty means backend handles caching
                model.snapshot_path.to_string_lossy().as_ref(),
                device,
                tensor_inputs,
            )
            .map_err(RuntimeError::Client)?;

        // Load output tensors
        let mut outputs = Vec::new();
        for output in result.outputs {
            let tensor = hdt::load(&output.path)
                .map_err(|e| RuntimeError::Other(format!("Failed to load output tensor: {}", e)))?;
            outputs.push((output.name, tensor));
        }

        Ok(outputs)
    }

    /// Get mutable reference to backend manager (for advanced use)
    pub fn backend_manager(&mut self) -> &mut backend::PluginManager {
        &mut self.backend_manager
    }

    /// Get mutable reference to format manager (for advanced use)
    pub fn format_manager(&mut self) -> &mut format::PluginManager {
        &mut self.format_manager
    }
}

/// A loaded model ready for execution
pub struct Model {
    snapshot_path: PathBuf,
    #[allow(dead_code)]
    cache_dir: PathBuf,
    #[allow(dead_code)]
    is_temp: bool,
}

impl Model {
    /// Get the snapshot path
    pub fn snapshot_path(&self) -> &Path {
        &self.snapshot_path
    }
}

/// Runtime errors
#[derive(Debug)]
pub enum RuntimeError {
    Format(format::ManagerError),
    Backend(backend::ManagerError),
    Client(ClientError),
    FileNotFound(String),
    Io(String),
    Other(String),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::Format(e) => write!(f, "Format plugin error: {}", e),
            RuntimeError::Backend(e) => write!(f, "Backend plugin error: {}", e),
            RuntimeError::Client(e) => write!(f, "Plugin client error: {}", e),
            RuntimeError::FileNotFound(path) => write!(f, "File not found: {}", path),
            RuntimeError::Io(e) => write!(f, "IO error: {}", e),
            RuntimeError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for RuntimeError {}

// Helper functions for dtype conversion
#[allow(dead_code)]
fn core_dtype_to_plugin(dtype: DType) -> PluginDType {
    match dtype {
        DType::BOOL => PluginDType::BOOL,
        DType::F8E4M3 => PluginDType::F8E4M3,
        DType::F8E5M2 => PluginDType::F8E5M2,
        DType::BF16 => PluginDType::BF16,
        DType::F16 => PluginDType::F16,
        DType::F32 => PluginDType::F32,
        DType::F64 => PluginDType::F64,
        DType::U8 => PluginDType::U8,
        DType::U16 => PluginDType::U16,
        DType::U32 => PluginDType::U32,
        DType::U64 => PluginDType::U64,
        DType::I8 => PluginDType::I8,
        DType::I16 => PluginDType::I16,
        DType::I32 => PluginDType::I32,
        DType::I64 => PluginDType::I64,
    }
}

#[allow(dead_code)]
fn plugin_dtype_to_core(dtype: PluginDType) -> DType {
    match dtype {
        PluginDType::BOOL => DType::BOOL,
        PluginDType::F8E4M3 => DType::F8E4M3,
        PluginDType::F8E5M2 => DType::F8E5M2,
        PluginDType::BF16 => DType::BF16,
        PluginDType::F16 => DType::F16,
        PluginDType::F32 => DType::F32,
        PluginDType::F64 => DType::F64,
        PluginDType::U8 => DType::U8,
        PluginDType::U16 => DType::U16,
        PluginDType::U32 => DType::U32,
        PluginDType::U64 => DType::U64,
        PluginDType::I8 => DType::I8,
        PluginDType::I16 => DType::I16,
        PluginDType::I32 => DType::I32,
        PluginDType::I64 => DType::I64,
        _ => DType::F32, // fallback for unknown types
    }
}
