//! Prelude module for convenient imports
//!
//! Usage: `use hodu_nn::prelude::*;`

// Re-export
pub use crate::losses::{
    binary_cross_entropy::BCEWithLogitsLoss, cosine_embedding::CosineEmbeddingLoss, cross_entropy::CrossEntropyLoss,
    kl_div::KLDivLoss, mae::MAELoss, mse::MSELoss, smooth_l1::SmoothL1Loss,
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
pub use crate::optimizers::{
    adagrad::Adagrad,
    adam::{Adam, AdamW},
    rmsprop::RMSprop,
    sgd::SGD,
};
pub use crate::{eval, train};
