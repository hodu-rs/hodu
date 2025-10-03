//! Prelude module for convenient imports
//!
//! Usage: `use hodu_nn::prelude::*;`

// Re-export core library types and functions
pub use crate::losses::{huber::Huber, mae::MAE, mse::MSE};
pub use crate::module::Module;
pub use crate::modules::{activation::*, linear::Linear};
pub use crate::optimizer::Optimizer;
pub use crate::optimizers::{adam::Adam, sgd::SGD};
