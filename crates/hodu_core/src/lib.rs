#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::result_large_err)]

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
pub mod format;
pub(crate) mod into;
pub mod op_metadatas;
pub mod op_params;
pub mod ops;
pub mod prelude;
pub mod scalar;
pub mod snapshot;
pub mod tensor;
pub mod types;
pub(crate) mod utils;

pub(crate) use hodu_compat as compat;
