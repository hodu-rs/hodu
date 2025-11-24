use crate::compat::*;
use crate::error::HoduResult;
use crate::tensor::Tensor;
use core::mem::ManuallyDrop;
use inkwell::{context::Context, execution_engine::ExecutionEngine};

/// LLVM-compiled state holding context and JIT execution engine
/// Uses 'static lifetime via Box::leak pattern for self-referential structure
pub struct LLVMJitState {
    /// Leaked context pointer - will be reclaimed on drop
    context_ptr: *mut Context,
    /// JIT execution engine with 'static lifetime (actually references context_ptr)
    engine: ManuallyDrop<ExecutionEngine<'static>>,
}

impl LLVMJitState {
    /// Create new LLVM JIT state
    /// SAFETY: Caller must ensure context and engine lifetimes are properly managed
    pub unsafe fn new(context: Context, engine: ExecutionEngine<'_>) -> Self {
        // Leak context to get 'static lifetime
        let context_ptr = Box::into_raw(Box::new(context));

        // Transmute engine to 'static (it actually references context_ptr)
        let engine: ExecutionEngine<'static> = core::mem::transmute(engine);

        Self {
            context_ptr,
            engine: ManuallyDrop::new(engine),
        }
    }

    /// Execute the compiled function with inputs and return outputs
    pub fn execute(&self, _inputs: &[(&str, &Tensor)]) -> HoduResult<HashMap<String, Tensor>> {
        // TODO: Implement actual execution
        // 1. Get function from engine
        // 2. Prepare input buffers from tensors
        // 3. Prepare output buffers
        // 4. Call JIT function
        // 5. Convert output buffers to tensors
        todo!("LLVM JIT execution not yet implemented")
    }
}

impl Drop for LLVMJitState {
    fn drop(&mut self) {
        // SAFETY: Drop engine first (which references context), then reclaim context
        unsafe {
            ManuallyDrop::drop(&mut self.engine);
            let _ = Box::from_raw(self.context_ptr);
        }
    }
}
