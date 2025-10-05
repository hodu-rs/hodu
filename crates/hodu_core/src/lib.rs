#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod backends;
pub mod compat;
pub mod error;
pub(crate) mod flatten;
pub mod prelude;
pub mod scalar;
pub mod tensor;
pub mod types;
