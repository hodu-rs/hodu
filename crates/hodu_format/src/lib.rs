//! Model and output format support for Hodu
//!
//! This crate provides format definitions and loaders for various formats:
//!
//! ## Graph formats (model serialization)
//! - **hdss**: Hodu Snapshot format (native serialized computation graph)
//! - **onnx**: ONNX format (planned)
//! - **gguf**: GGUF format (planned)
//! - **pytorch**: PyTorch format (planned)
//!
//! ## Tensor formats (input/output data)
//! - **hdt**: Hodu Tensor format (native binary tensor)
//! - **json**: JSON tensor format (human-readable, debugging)
//! - **safetensors**: SafeTensors format (planned)
//! - **npy**: NumPy format (planned)

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

mod format;

// Graph formats
#[cfg(feature = "hdss")]
pub mod hdss;

// Tensor formats
#[cfg(feature = "hdt")]
pub mod hdt;

#[cfg(feature = "json")]
pub mod json;

pub use format::*;
