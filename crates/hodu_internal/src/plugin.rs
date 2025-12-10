//! Plugin runtime for loading and running models
//!
//! Enable with the `plugin` feature flag.
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
//! let outputs = runtime.run(&model, &[("input", &input)], "cpu")?;
//! ```

// High-level API
pub use hodu_plugin_runtime::{Model, Runtime, RuntimeError};

// Low-level API for advanced use
pub use hodu_plugin_runtime::backend::{ManagerError as BackendManagerError, PluginManager as BackendPluginManager};
pub use hodu_plugin_runtime::format::{ManagerError as FormatManagerError, PluginManager as FormatPluginManager};
pub use hodu_plugin_runtime::{ClientError, PluginClient};
