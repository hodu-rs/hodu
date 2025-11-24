pub mod hodu;
#[cfg(feature = "xla")]
pub mod xla;

use crate::compat::*;
use crate::error::HoduResult;
use crate::tensor::Tensor;
pub use hodu::{HoduCompiledState, LLVMJitState};

#[cfg(feature = "xla")]
pub use xla::XLAExecutable;

/// Runtime-specific compiled state
pub enum CompiledState {
    /// HODU runtime with compiler backend
    HODU(HoduCompiledState),

    /// XLA runtime
    #[cfg(feature = "xla")]
    XLA(XLAExecutable),
    // Future: ONNX, TVM, etc.
}

impl CompiledState {
    /// Execute the compiled script with inputs
    pub fn execute(&self, inputs: &[(&str, &Tensor)]) -> HoduResult<HashMap<String, Tensor>> {
        match self {
            CompiledState::HODU(state) => state.execute(inputs),
            #[cfg(feature = "xla")]
            CompiledState::XLA(state) => state.execute(inputs),
        }
    }
}
