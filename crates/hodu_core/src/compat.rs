//! std/no-std compatibility layer
//!
//! Internal module for handling differences between std and no-std environments.

// Basic types and formatting
#[cfg(not(feature = "std"))]
pub use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

#[cfg(feature = "std")]
pub use std::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

// Collections
#[cfg(not(feature = "std"))]
pub use alloc::collections::{BTreeMap as HashMap, BTreeSet as HashSet};

#[cfg(feature = "std")]
pub use std::collections::{HashMap, HashSet};

// Synchronization primitives
#[cfg(not(feature = "std"))]
pub use alloc::sync::Arc;

#[cfg(feature = "std")]
pub use std::sync::Arc;

#[cfg(not(feature = "std"))]
pub use spin::{Lazy as LazyLock, Mutex, RwLock};

#[cfg(feature = "std")]
pub use std::sync::{LazyLock, Mutex, RwLock};

// Atomic operations
#[cfg(not(feature = "std"))]
pub use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[cfg(feature = "std")]
pub use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

// Core traits and functions  
pub use core::fmt;
