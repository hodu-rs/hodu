//! Error types for CPU kernel operations

#[cfg(not(feature = "std"))]
use alloc::string::String;
use core::fmt;

/// Error type for CPU kernel operations
#[derive(Debug, Clone)]
pub enum CpuKernelError {
    /// Invalid kernel name or configuration
    InvalidKernel(String),
    /// Invalid input parameters or dimensions
    InvalidInput(String),
    /// Null pointer error
    NullPointer(String),
    /// Generic error message
    Message(String),
}

impl fmt::Display for CpuKernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CpuKernelError::InvalidKernel(msg) => write!(f, "invalid kernel: {}", msg),
            CpuKernelError::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
            CpuKernelError::NullPointer(msg) => write!(f, "null pointer: {}", msg),
            CpuKernelError::Message(msg) => write!(f, "message: {}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CpuKernelError {}

/// Result type for CPU kernel operations
pub type Result<T> = core::result::Result<T, CpuKernelError>;
