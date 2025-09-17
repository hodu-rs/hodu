#![no_std]
#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod backends;
pub(crate) mod compat;
pub mod error;
pub(crate) mod flatten;
pub mod prelude;
pub mod scalar;
pub mod tensor;
pub mod types;
