//! Hodu CPU Kernels
//!
//! High-performance CPU kernels for tensor operations with support for
//! exotic floating-point formats and comprehensive integer types.

#![cfg_attr(not(feature = "std"), no_std)]

/// Path to the C kernel source files directory.
/// Useful for AOT compilation backends that need to compile kernels.
pub const KERNELS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels");

#[cfg(not(feature = "std"))]
extern crate alloc;

mod error;
pub mod jit_symbols;
mod kernels;

pub use error::{CpuKernelError, Result};
pub use kernels::*;
