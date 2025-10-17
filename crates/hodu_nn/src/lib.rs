#![cfg_attr(not(feature = "std"), no_std)]

pub(crate) mod compat;
mod losses;
pub mod module;
mod modules;
pub mod optimizer;
mod optimizers;
pub mod prelude;

pub use losses::{
    binary_cross_entropy::{BCELoss, BCEWithLogitsLoss},
    cross_entropy::CrossEntropyLoss,
    huber::Huber,
    mae::MAE,
    mse::MSE,
    nll::NLLLoss,
};
pub use modules::{
    activation::*,
    conv::{Conv1D, Conv2D, Conv3D, ConvTranspose1D, ConvTranspose2D, ConvTranspose3D},
    linear::Linear,
};
pub use optimizers::{
    adam::{Adam, AdamW},
    sgd::SGD,
};
