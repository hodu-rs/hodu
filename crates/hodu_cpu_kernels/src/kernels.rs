#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::too_many_arguments)]

pub mod binary;
pub mod concat_split;
pub mod conv;
pub mod indexing;
pub mod macros;
pub mod matrix;
pub mod reduce;
pub mod unary;
pub mod windowing;

// Re-export the Kernel type
pub use macros::Kernel;

// Re-export all operations
pub use binary::*;
pub use concat_split::*;
pub use conv::*;
pub use indexing::*;
pub use matrix::*;
pub use reduce::*;
pub use unary::*;
pub use windowing::*;
