//! Hodu plugin runtime
//!
//! This crate provides the runtime for loading and executing Hodu plugins.
//! It supports both format plugins (for model/tensor file formats) and
//! backend plugins (for inference execution).
//!
//! # Feature flags
//!
//! - `format` - Enable format plugin support (load/save models and tensors)
//! - `backend` - Enable backend plugin support (run inference, build models)
//!
//! # Example
//!
//! ```ignore
//! use hodu_plugin_runtime::format;
//!
//! // Load a model using a format plugin
//! let mut manager = format::PluginManager::new()?;
//! let client = manager.get_format_for_extension("onnx")?;
//! let model = client.load_model("model.onnx")?;
//! ```

#[cfg(feature = "backend")]
pub mod backend;
mod client;
#[cfg(feature = "format")]
pub mod format;
mod registry;
#[cfg(all(feature = "format", feature = "backend"))]
mod runtime;
mod types;

pub use client::{CancellationHandle, ClientError, PluginClient, DEFAULT_TIMEOUT};
pub use registry::{detect_plugin_type, PluginDetectError, PluginRegistry, RegistryError};
#[cfg(all(feature = "format", feature = "backend"))]
pub use runtime::{Model, Runtime, RuntimeError};
pub use types::{DetectedPluginType, PluginCapabilities, PluginEntry, PluginSource, PluginType};
