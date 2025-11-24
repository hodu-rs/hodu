pub mod llvm;

use crate::compat::*;
use crate::error::HoduResult;
use crate::tensor::Tensor;
pub use llvm::LLVMJitState;

/// HODU runtime compiler-specific compiled state
pub enum HoduCompiledState {
    /// LLVM JIT compiled state
    LLVM(LLVMJitState),
    // Future: MLIR, Cranelift, etc.
}

impl HoduCompiledState {
    /// Execute the compiled function with inputs
    pub fn execute(&self, inputs: &[(&str, &Tensor)]) -> HoduResult<HashMap<String, Tensor>> {
        match self {
            HoduCompiledState::LLVM(state) => state.execute(inputs),
        }
    }
}
