//! Prelude module for convenient imports
//!
//! Usage: `use hodu_nn::prelude::*;`

// Re-export core library types and functions
pub use crate::losses::{
    binary_cross_entropy::BCEWithLogitsLoss, cross_entropy::CrossEntropyLoss, mae::MAELoss, mse::MSELoss,
};
pub use crate::module::Module;
pub use crate::modules::{
    activation::*,
    conv::Conv2D,
    dropout::Dropout,
    linear::Linear,
    norm::{BatchNorm1D, BatchNorm2D, BatchNorm3D, LayerNorm},
    pooling::{AdaptiveAvgPool2D, AdaptiveMaxPool2D, AvgPool2D, MaxPool2D},
};
pub use crate::optimizer::Optimizer;
pub use crate::optimizers::adam::{Adam, AdamW};
pub use crate::{eval, train};
