#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub(crate) mod be;
pub(crate) mod be_cpu;
#[cfg(feature = "cuda")]
pub(crate) mod be_cuda;
#[cfg(feature = "metal")]
pub(crate) mod be_metal;
pub(crate) mod cache;
pub mod error;
pub(crate) mod into;
pub mod layer;
pub mod ops;
pub mod prelude;
pub mod scalar;
pub mod script;
pub mod tensor;
pub mod types;
pub(crate) mod utils;
