#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod compat;
pub mod cuda;
pub mod error;
pub mod kernel;
pub mod kernels;
pub mod source;
pub use cudarc;

pub use cuda::*;
pub use kernels::macros::Kernel;
