//! Gradient tape management
//!
//! This module handles the recording and cleanup of operations on the gradient tape.

use super::context::ContextId;
use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{Op, OpParams},
    tensor::TensorId,
};

/// A single entry in the gradient tape
#[derive(Clone)]
pub(super) struct TapeEntry {
    pub output_id: TensorId,
    pub op: Op,
    pub op_params: OpParams,
    pub input_ids: Vec<TensorId>,
}

/// Global gradient tape storage (one tape per context)
#[cfg(feature = "std")]
static GRADIENT_TAPES: LazyLock<Mutex<HashMap<ContextId, Vec<TapeEntry>>>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    map.insert(ContextId::DEFAULT, Vec::new()); // Default context
    Mutex::new(map)
});

#[cfg(not(feature = "std"))]
static GRADIENT_TAPES: LazyLock<Mutex<HashMap<ContextId, Vec<TapeEntry>>>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    map.insert(ContextId::DEFAULT, Vec::new()); // Default context
    Mutex::new(map)
});

/// Initialize a new tape for a context
pub(super) fn initialize_tape(context_id: ContextId) {
    #[cfg(feature = "std")]
    {
        GRADIENT_TAPES.lock().unwrap().insert(context_id, Vec::new());
    }
    #[cfg(not(feature = "std"))]
    {
        GRADIENT_TAPES.lock().insert(context_id, Vec::new());
    }
}

/// Remove a tape for a context and clean up context-owned tensors
pub(super) fn remove_tape(context_id: ContextId) {
    #[cfg(feature = "std")]
    {
        if let Some(tape) = GRADIENT_TAPES.lock().unwrap().remove(&context_id) {
            cleanup_context_tensors(context_id, &tape);
        }
    }
    #[cfg(not(feature = "std"))]
    {
        if let Some(tape) = GRADIENT_TAPES.lock().remove(&context_id) {
            cleanup_context_tensors(context_id, &tape);
        }
    }
}

/// Clean up tensors owned by a specific context
fn cleanup_context_tensors(context_id: ContextId, _tape: &[TapeEntry]) {
    use crate::layer::compat::Ordering;
    use crate::tensor;

    // Collect ALL tensor IDs owned by this context with ref_count=0
    // (not just from tape, because some intermediate tensors may not be on the tape)
    let tensor_ids_to_check = tensor::get_all_tensor_ids();

    // Remove tensors that belong to this context AND have ref_count=0
    for tensor_id in tensor_ids_to_check {
        if let Some((should_remove, grad_id)) = tensor::with_tensor(tensor_id, |t| {
            // Remove if:
            // 1. (Owned by this context OR owner_context==None) AND ref_count==0
            // 2. NOT a gradient tensor (gradient tensors should be preserved)
            let ref_count = t.ref_count.load(Ordering::Relaxed);
            let owner_ctx = t.owner_context();
            let should_remove = (owner_ctx == Some(context_id) || owner_ctx.is_none())
                && ref_count == 0
                && t.is_runtime
                && !t.is_gradient;
            (should_remove, t.grad_tensor_id)
        }) {
            if should_remove {
                // If this tensor has a gradient, decrement its ref_count and remove if necessary
                if let Some(grad_id) = grad_id {
                    if let Some((should_remove_grad, should_cleanup_tape)) = tensor::with_tensor(grad_id, |g| {
                        let prev_count = g.ref_count.load(Ordering::Relaxed);
                        if prev_count > 0 {
                            g.ref_count.fetch_sub(1, Ordering::Relaxed);
                        }
                        let remove = (prev_count == 1 || prev_count == 0)
                            && g.is_runtime
                            && g.owner_context.is_none()
                            && !g.is_gradient;
                        let cleanup =
                            (prev_count == 1 || prev_count == 0) && g.requires_grad && g.owner_context.is_none();
                        (remove, cleanup)
                    }) {
                        if should_cleanup_tape {
                            remove_entries_with_input(context_id, grad_id);
                        }
                        if should_remove_grad {
                            tensor::remove(grad_id);
                        }
                    }
                }

                tensor::remove(tensor_id);
            }
        }
    }
}

/// Clean up tensors owned by the default context after backward
/// Only removes tensors with ref_count=0
pub(super) fn cleanup_default_context_after_backward() {
    let context_id = ContextId::DEFAULT;
    let tape = get_tape_clone(context_id).unwrap_or_default();
    cleanup_context_tensors(context_id, &tape);
}

/// Add an entry to the tape for a given context
pub(super) fn push_entry(context_id: ContextId, entry: TapeEntry) -> HoduResult<()> {
    #[cfg(feature = "std")]
    {
        let mut tapes = GRADIENT_TAPES.lock().map_err(|_| HoduError::GradientTapeCorrupted)?;
        if let Some(tape) = tapes.get_mut(&context_id) {
            tape.push(entry);
        }
    }
    #[cfg(not(feature = "std"))]
    {
        let mut tapes = GRADIENT_TAPES.lock();
        if let Some(tape) = tapes.get_mut(&context_id) {
            tape.push(entry);
        }
    }
    Ok(())
}

/// Get a cloned copy of the tape for a given context
pub(super) fn get_tape_clone(context_id: ContextId) -> HoduResult<Vec<TapeEntry>> {
    #[cfg(feature = "std")]
    {
        let tapes = GRADIENT_TAPES.lock().map_err(|_| HoduError::GradientTapeCorrupted)?;
        Ok(tapes.get(&context_id).cloned().unwrap_or_default())
    }
    #[cfg(not(feature = "std"))]
    {
        let tapes = GRADIENT_TAPES.lock();
        Ok(tapes.get(&context_id).cloned().unwrap_or_default())
    }
}

/// Clear the tape for a given context (but don't remove tensors)
pub(super) fn clear_tape_for_context(context_id: ContextId) {
    #[cfg(feature = "std")]
    {
        if let Ok(mut tapes) = GRADIENT_TAPES.lock() {
            if let Some(tape) = tapes.get_mut(&context_id) {
                tape.clear();
            }
        }
    }
    #[cfg(not(feature = "std"))]
    {
        let mut tapes = GRADIENT_TAPES.lock();
        if let Some(tape) = tapes.get_mut(&context_id) {
            tape.clear();
        }
    }
}

/// Remove tape entries that have the given tensor as input
/// This is called when a tensor with requires_grad=true is dropped
pub(super) fn remove_entries_with_input(context_id: ContextId, tensor_id: TensorId) {
    #[cfg(feature = "std")]
    {
        if let Ok(mut tapes) = GRADIENT_TAPES.lock() {
            if let Some(tape) = tapes.get_mut(&context_id) {
                // Remove entries where tensor_id is an input
                tape.retain(|entry| !entry.input_ids.contains(&tensor_id));
            }
        }
    }
    #[cfg(not(feature = "std"))]
    {
        let mut tapes = GRADIENT_TAPES.lock();
        if let Some(tape) = tapes.get_mut(&context_id) {
            tape.retain(|entry| !entry.input_ids.contains(&tensor_id));
        }
    }
}
