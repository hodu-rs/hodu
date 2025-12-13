//! Gradient computation and automatic differentiation
//!
//! This module provides automatic differentiation via reverse-mode gradient computation.
//! It consists of several submodules:
//!
//! - `core`: Core types (ContextId) and global flags
//! - `compute`: Gradient computation engine (compute_gradients, record_operation)
//! - `context`: Gradient context management (GradientContext RAII guard)
//! - `tape`: Tape recording and cleanup for computational graph
//! - `vjp`: VJP (Vector-Jacobian Product) trait definition
//! - `vjp_*`: VJP implementations for each operation type

mod compute;
mod context;
mod core;
mod tape;
mod vjp;

mod vjp_binary;
mod vjp_cmp;
mod vjp_concat_split;
mod vjp_conv;
mod vjp_einsum;
mod vjp_indexing;
mod vjp_linalg;
mod vjp_matrix;
mod vjp_padding;
mod vjp_reduce;
mod vjp_resize;
mod vjp_scan;
mod vjp_shape;
mod vjp_shape_memory;
mod vjp_sort;
mod vjp_unary;
mod vjp_utils;
mod vjp_windowing;

pub use compute::{compute_gradients, record_operation};
pub use context::GradientContext;
pub use core::{is_computing_gradients, is_in_optimizer_step, set_optimizer_step_flag, ContextId};

pub(crate) use core::get_active_context;
pub(crate) use vjp::VjpCompute;

pub(crate) fn cleanup_tape_for_dropped_tensor(tensor_id: crate::tensor::TensorId) {
    let context_id = get_active_context();
    tape::remove_entries_with_input(context_id, tensor_id);
}
