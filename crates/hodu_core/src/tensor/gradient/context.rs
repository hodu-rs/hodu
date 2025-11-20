use super::core::{pop_context, push_context, ContextId};

pub struct GradientContext {
    context_id: ContextId,
}

impl GradientContext {
    pub fn new() -> Self {
        let context_id = ContextId::new();

        push_context(context_id);

        super::tape::initialize_tape(context_id);

        Self { context_id }
    }

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
        if self.context_id == ContextId::DEFAULT {
            return;
        }

        pop_context();

        super::tape::remove_tape(self.context_id);
    }
}
