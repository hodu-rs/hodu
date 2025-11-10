//! std/no_std compatibility layer
//!
//! Internal module for handling differences between std and no-std environments,
//! and cudarc API compatibility.

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
pub use alloc::sync::{Arc, Weak};

#[cfg(feature = "std")]
pub use std::sync::{Arc, Weak};

#[cfg(not(feature = "std"))]
pub use spin::{Mutex, RwLock};

#[cfg(feature = "std")]
pub use std::sync::{Mutex, RwLock};

// Core traits and functions
#[cfg(not(feature = "std"))]
pub use core::{
    cell::{Cell, RefCell},
    fmt,
    hash::{Hash, Hasher},
    ops,
};

#[cfg(feature = "std")]
pub use std::{
    cell::{Cell, RefCell},
    fmt,
    hash::{Hash, Hasher},
    ops,
};

// Lazy/OnceLock compatibility
#[cfg(not(feature = "std"))]
pub use spin::Lazy;

#[cfg(feature = "std")]
pub use std::sync::LazyLock as Lazy;

// RwLock compatibility wrapper
pub struct RwLockGuard<T>(T);

impl<T> core::ops::Deref for RwLockGuard<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> core::ops::DerefMut for RwLockGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// Extension trait for RwLock to provide Result-returning API
pub trait RwLockExt<T> {
    fn read_compat(&self) -> Result<RwLockGuard<impl core::ops::Deref<Target = T> + '_>, String>;
    fn write_compat(&self) -> Result<RwLockGuard<impl core::ops::DerefMut<Target = T> + '_>, String>;
}

#[cfg(feature = "std")]
impl<T> RwLockExt<T> for RwLock<T> {
    fn read_compat(&self) -> Result<RwLockGuard<impl core::ops::Deref<Target = T> + '_>, String> {
        self.read().map(RwLockGuard).map_err(|e| format!("{:?}", e))
    }

    fn write_compat(&self) -> Result<RwLockGuard<impl core::ops::DerefMut<Target = T> + '_>, String> {
        self.write().map(RwLockGuard).map_err(|e| format!("{:?}", e))
    }
}

#[cfg(not(feature = "std"))]
impl<T> RwLockExt<T> for RwLock<T> {
    fn read_compat(&self) -> Result<RwLockGuard<impl core::ops::Deref<Target = T> + '_>, String> {
        Ok(RwLockGuard(self.read()))
    }

    fn write_compat(&self) -> Result<RwLockGuard<impl core::ops::DerefMut<Target = T> + '_>, String> {
        Ok(RwLockGuard(self.write()))
    }
}
