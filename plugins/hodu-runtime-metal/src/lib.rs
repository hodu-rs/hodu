//! Metal Runtime Plugin for Hodu
//!
//! Executes compiled Metal artifacts on Apple GPUs.

mod dispatch;
mod runtime;

pub use runtime::MetalRuntime;
