use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ContextId(usize);

impl ContextId {
    pub const DEFAULT: ContextId = ContextId(0);

    pub(super) fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

thread_local! {
    static CONTEXT_STACK: RefCell<Vec<ContextId>> = const { RefCell::new(vec![]) };
}

pub(crate) fn get_active_context() -> ContextId {
    CONTEXT_STACK.with(|stack| stack.borrow().last().copied().unwrap_or(ContextId::DEFAULT))
}

pub(super) fn push_context(context_id: ContextId) {
    CONTEXT_STACK.with(|stack| {
        stack.borrow_mut().push(context_id);
    });
}

pub(super) fn pop_context() {
    CONTEXT_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });
}

static IS_COMPUTING_GRADIENTS: AtomicBool = AtomicBool::new(false);
static IS_IN_OPTIMIZER_STEP: AtomicBool = AtomicBool::new(false);

pub fn is_computing_gradients() -> bool {
    IS_COMPUTING_GRADIENTS.load(Ordering::Relaxed)
}

pub fn is_in_optimizer_step() -> bool {
    IS_IN_OPTIMIZER_STEP.load(Ordering::Relaxed)
}

pub fn set_optimizer_step_flag(value: bool) {
    IS_IN_OPTIMIZER_STEP.store(value, Ordering::Relaxed);
}

pub(super) fn set_computing_gradients(value: bool) {
    IS_COMPUTING_GRADIENTS.store(value, Ordering::Relaxed);
}
