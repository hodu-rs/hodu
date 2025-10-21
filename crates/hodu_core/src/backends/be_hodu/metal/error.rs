use crate::types::dtype::DType;
use std::error::Error;
use std::fmt;
use std::sync::{PoisonError, TryLockError};

#[derive(Debug)]
pub enum LockError {
    Poisoned(String),
    WouldBlock,
}

impl fmt::Display for LockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LockError::Poisoned(msg) => write!(f, "{}", msg),
            LockError::WouldBlock => write!(f, "Would block"),
        }
    }
}

impl Error for LockError {}

impl<T> From<TryLockError<T>> for MetalError {
    fn from(value: TryLockError<T>) -> Self {
        match value {
            TryLockError::Poisoned(p) => MetalError::LockError(LockError::Poisoned(p.to_string())),
            TryLockError::WouldBlock => MetalError::LockError(LockError::WouldBlock),
        }
    }
}

impl<T> From<PoisonError<T>> for MetalError {
    fn from(p: PoisonError<T>) -> Self {
        MetalError::LockError(LockError::Poisoned(p.to_string()))
    }
}

#[derive(Debug)]
pub enum MetalError {
    Message(String),
    KernelError(hodu_metal_kernels::error::MetalKernelError),
    LockError(LockError),
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalError::Message(msg) => write!(f, "{}", msg),
            MetalError::KernelError(e) => write!(f, "{}", e),
            MetalError::LockError(e) => write!(f, "{:?}", e),
            MetalError::UnexpectedDType { msg, expected, got } => {
                write!(f, "{}, expected: {:?}, got: {:?}", msg, expected, got)
            },
        }
    }
}

impl Error for MetalError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            MetalError::KernelError(e) => Some(e),
            MetalError::LockError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<hodu_metal_kernels::error::MetalKernelError> for MetalError {
    fn from(e: hodu_metal_kernels::error::MetalKernelError) -> Self {
        MetalError::KernelError(e)
    }
}

impl From<String> for MetalError {
    fn from(e: String) -> Self {
        MetalError::Message(e)
    }
}
