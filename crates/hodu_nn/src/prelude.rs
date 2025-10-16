//! Prelude module for convenient imports
//!
//! Usage: `use hodu_nn::prelude::*;`

// Re-export core library types and functions
pub use crate::losses::{
    binary_cross_entropy::{BCELoss, BCEWithLogitsLoss},
    cross_entropy::CrossEntropyLoss,
    huber::Huber,
    mae::MAE,
    mse::MSE,
    nll::NLLLoss,
};
pub use crate::module::Module;
pub use crate::modules::{activation::*, linear::Linear};
pub use crate::optimizer::Optimizer;
pub use crate::optimizers::{adam::Adam, sgd::SGD};
