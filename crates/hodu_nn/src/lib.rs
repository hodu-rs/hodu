#![cfg_attr(not(feature = "std"), no_std)]

pub(crate) mod compat;
mod losses;
pub mod module;
mod modules;
pub mod optimizer;
mod optimizers;
pub mod prelude;

pub use losses::{huber::Huber, mae::MAE, mse::MSE};
pub use modules::{activation::*, linear::Linear};
pub use optimizers::{adam::Adam, sgd::SGD};
