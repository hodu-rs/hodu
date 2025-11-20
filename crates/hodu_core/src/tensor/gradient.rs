//! Gradient computation and automatic differentiation
//!
//! This module provides automatic differentiation via reverse-mode gradient computation.
//! It consists of several submodules:
//!
//! - `backprop`: Gradient computation engine (backward pass)
//! - `context`: Gradient context management for nested scopes
//! - `tape`: Tape recording and cleanup for computational graph
//! - `vjp`: VJP (Vector-Jacobian Product) trait definition
//! - `vjp_*`: VJP implementations for each operation type

mod backprop;
mod context;
mod tape;
mod vjp;

// VJP implementations for different operation types
mod vjp_binary;
mod vjp_cmp;
mod vjp_concat_split;
mod vjp_conv;
mod vjp_indexing;
mod vjp_matrix;
mod vjp_reduce;
mod vjp_shape;
mod vjp_unary;
mod vjp_utils;
mod vjp_windowing;

// Re-export public APIs
pub use backprop::{
    compute_gradients, is_computing_gradients, is_in_optimizer_step, record_operation, record_operation_with_dims,
    record_operation_with_scalar, record_operation_with_scalars, record_operation_with_split_info,
    set_optimizer_step_flag,
};
pub use context::{ContextId, GradientContext};

// Internal use only
pub(crate) use context::get_active_context;
pub(crate) use vjp::VjpCompute;

// Tape cleanup (called from Tensor::drop)
pub(crate) fn cleanup_tape_for_dropped_tensor(tensor_id: crate::tensor::TensorId) {
    let context_id = get_active_context();
    tape::remove_entries_with_input(context_id, tensor_id);
}
