//! Gradient context management
//!
//! This module manages gradient computation contexts for nested scopes and
//! tracks which tensors belong to which context for memory cleanup.

use crate::layer::compat::*;

/// Unique identifier for gradient contexts
///
/// Each gradient context has a unique ID. The default context has ID 0.
/// User-created tensors have owner_context = None, while tensors created
/// during operations within a context have owner_context = Some(ContextId).
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ContextId(usize);

impl ContextId {
    /// Default context (always exists)
    pub const DEFAULT: ContextId = ContextId(0);

    /// Create new unique context ID
    pub(super) fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(1); // Start from 1 (0 is default)
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

// Context stack for nested gradient contexts
#[cfg(feature = "std")]
thread_local! {
    static CONTEXT_STACK: RefCell<Vec<ContextId>> = const { RefCell::new(vec![]) };
}

#[cfg(not(feature = "std"))]
static CONTEXT_STACK: Mutex<Vec<ContextId>> = Mutex::new(Vec::new());

/// Get the currently active context ID
pub(crate) fn get_active_context() -> ContextId {
    #[cfg(feature = "std")]
    {
        CONTEXT_STACK.with(|stack| stack.borrow().last().copied().unwrap_or(ContextId::DEFAULT))
    }
    #[cfg(not(feature = "std"))]
    {
        CONTEXT_STACK.lock().last().copied().unwrap_or(ContextId::DEFAULT)
    }
}

/// Push a context onto the stack
pub(super) fn push_context(context_id: ContextId) {
    #[cfg(feature = "std")]
    {
        CONTEXT_STACK.with(|stack| {
            stack.borrow_mut().push(context_id);
        });
    }
    #[cfg(not(feature = "std"))]
    {
        CONTEXT_STACK.lock().push(context_id);
    }
}

/// Pop a context from the stack
pub(super) fn pop_context() {
    #[cfg(feature = "std")]
    {
        CONTEXT_STACK.with(|stack| {
            stack.borrow_mut().pop();
        });
    }
    #[cfg(not(feature = "std"))]
    {
        CONTEXT_STACK.lock().pop();
    }
}

/// RAII guard for gradient context management
///
/// When dropped, automatically cleans up the context and its tape.
pub struct GradientContext {
    context_id: ContextId,
}

impl GradientContext {
    /// Create a new gradient context
    pub fn new() -> Self {
        let context_id = ContextId::new();

        // Push to context stack
        push_context(context_id);

        // Initialize empty tape
        super::tape::initialize_tape(context_id);

        Self { context_id }
    }

    /// Get the context ID
    pub fn id(&self) -> ContextId {
        self.context_id
    }
}

impl Default for GradientContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for GradientContext {
    fn drop(&mut self) {
        // Don't drop the default context (ID 0)
        if self.context_id == ContextId::DEFAULT {
            return;
        }

        // Pop from context stack
        pop_context();

        // Remove tape (this will also clean up context-owned tensors)
        super::tape::remove_tape(self.context_id);
    }
}
