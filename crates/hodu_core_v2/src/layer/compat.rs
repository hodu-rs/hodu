//! std/no-std compatibility layer
//!
//! Internal module for handling differences between std and no-std environments.

#![allow(unused_imports)]

// Basic types and formatting
#[cfg(not(feature = "std"))]
pub use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

#[cfg(feature = "std")]
pub use std::{
    boxed::Box,
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
pub use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};

#[cfg(feature = "std")]
pub use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};

// Core traits and functions
#[cfg(not(feature = "std"))]
pub use core::{
    any::TypeId,
    cell::{Cell, RefCell},
    fmt, ops,
};

#[cfg(feature = "std")]
pub use std::{
    any::TypeId,
    cell::{Cell, RefCell},
    fmt, ops, thread_local,
};

// Debug printing
#[cfg(feature = "std")]
pub use std::{eprintln, println};

// Parallel iteration
#[cfg(all(feature = "std", feature = "rayon"))]
pub use rayon::prelude::*;
