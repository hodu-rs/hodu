//! Prelude module for convenient imports
//!
//! Usage: `use hodu_nn::prelude::*;`

// Re-export core library types and functions
pub use crate::losses::{binary_cross_entropy::BCEWithLogitsLoss, cross_entropy::CrossEntropyLoss, mae::MAE, mse::MSE};
pub use crate::module::Module;
pub use crate::modules::{activation::*, conv::Conv2D, dropout::Dropout, linear::Linear};
pub use crate::optimizer::Optimizer;
pub use crate::optimizers::adam::{Adam, AdamW};
pub use crate::{eval, train};
