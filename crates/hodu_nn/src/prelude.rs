//! Prelude module for convenient imports
//!
//! Usage: `use hodu_nn::prelude::*;`

// Re-export
pub use crate::losses::{
    binary_cross_entropy::BCEWithLogitsLoss, cross_entropy::CrossEntropyLoss, mae::MAELoss, mse::MSELoss,
};
pub use crate::module::Module;
pub use crate::modules::{
    activation::*,
    attention::{scaled_dot_product_attention, MultiheadAttention},
    conv::Conv2D,
    dropout::Dropout,
    embedding::Embedding,
    linear::Linear,
    norm::{BatchNorm1D, BatchNorm2D, BatchNorm3D, LayerNorm},
    pooling::{AdaptiveAvgPool2D, AdaptiveMaxPool2D, AvgPool2D, MaxPool2D},
    rnn::{Nonlinearity, GRU, LSTM, RNN},
};
pub use crate::optimizer::Optimizer;
pub use crate::optimizers::adam::{Adam, AdamW};
pub use crate::{eval, train};
