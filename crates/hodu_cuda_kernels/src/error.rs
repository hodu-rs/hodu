//! Error types for CUDA kernel operations

use crate::compat::*;

/// Error type for CUDA kernel operations
#[derive(Debug, Clone)]
pub enum CudaKernelError {
    /// Invalid kernel name or configuration
    InvalidKernel(String),
    /// Invalid input parameters or dimensions
    InvalidInput(String),
    /// CUDA launch error
    LaunchError(String),
    /// CUDA memory error
    MemoryError(String),
    /// Generic error message
    Message(String),
}

impl fmt::Display for CudaKernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaKernelError::InvalidKernel(msg) => write!(f, "invalid kernel: {}", msg),
            CudaKernelError::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
            CudaKernelError::LaunchError(msg) => write!(f, "launch error: {}", msg),
            CudaKernelError::MemoryError(msg) => write!(f, "memory error: {}", msg),
            CudaKernelError::Message(msg) => write!(f, "message: {}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CudaKernelError {}

/// Result type for CUDA kernel operations
pub type Result<T> = core::result::Result<T, CudaKernelError>;
