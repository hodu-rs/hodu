#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::too_many_arguments)]

pub mod macros;
pub mod ops_binary;
pub mod ops_cast;
pub mod ops_concat_split;
pub mod ops_conv;
pub mod ops_einsum;
pub mod ops_indexing;
pub mod ops_matrix;
pub mod ops_memory;
pub mod ops_padding;
pub mod ops_reduce;
pub mod ops_resize;
pub mod ops_scan;
pub mod ops_shape_memory;
pub mod ops_sort;
pub mod ops_unary;
pub mod ops_windowing;
pub mod storage;

// Re-export the Kernel type
pub use macros::Kernel;

// Re-export all operations
pub use ops_binary::*;
pub use ops_cast::*;
pub use ops_concat_split::*;
pub use ops_conv::*;
pub use ops_einsum::*;
pub use ops_indexing::*;
pub use ops_matrix::*;
pub use ops_memory::*;
pub use ops_padding::*;
pub use ops_reduce::*;
pub use ops_resize::*;
pub use ops_scan::*;
pub use ops_shape_memory::*;
pub use ops_sort::*;
pub use ops_unary::*;
pub use ops_windowing::*;
pub use storage::*;
