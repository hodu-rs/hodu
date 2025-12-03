//! Plugin system for Hodu CLI
//!
//! This module provides the JSON-RPC based plugin system for loading and
//! communicating with format and backend plugins.

mod client;
mod process;
mod registry;

pub use process::*;
pub use registry::*;
