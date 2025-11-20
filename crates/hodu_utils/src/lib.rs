#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::result_large_err)]

pub mod data;
pub mod prelude;

pub(crate) use hodu_compat as compat;
