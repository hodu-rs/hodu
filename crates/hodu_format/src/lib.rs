//! Model and output format support for Hodu
//!
//! This crate provides format definitions and loaders for various model formats:
//! - **hdss**: Hodu Snapshot format (native serialized computation graph)
//! - **onnx**: ONNX format (planned)
//! - **safetensors**: SafeTensors format (planned)
//! - **gguf**: GGUF format (planned)
//! - **pytorch**: PyTorch format (planned)

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

mod format;

#[cfg(feature = "hdss")]
pub mod hdss;

pub use format::*;
