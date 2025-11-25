//! Hodu CPU Kernels
//!
//! High-performance CPU kernels for tensor operations with support for
//! exotic floating-point formats and comprehensive integer types.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

mod error;
pub mod jit_symbols;
mod kernels;

pub use error::{CpuKernelError, Result};
pub use kernels::*;
