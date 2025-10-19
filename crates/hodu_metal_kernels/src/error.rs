use std::fmt;

#[derive(Debug)]
pub enum MetalKernelError {
    CommandBufferError(String),
    LockError(String),
    LoadLibraryError(String),
    LoadFunctionError(String),
    UnsupportedDTypeForOp(&'static str, &'static str),
    FailedToCreateComputeFunction,
    FailedToCreateResource(String),
    FailedToCreatePipeline(String),
    MatMulNonContiguous {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
        mnk: (usize, usize, usize),
    },
    WithBacktrace {
        inner: Box<Self>,
        backtrace: Box<std::backtrace::Backtrace>,
    },
}

impl fmt::Display for MetalKernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CommandBufferError(msg) => {
                write!(f, "Command buffer had following error: {}", msg)
            },
            Self::LockError(msg) => {
                write!(f, "Could not lock resource: {}", msg)
            },
            Self::LoadLibraryError(msg) => {
                write!(f, "Error while loading library: {}", msg)
            },
            Self::LoadFunctionError(msg) => {
                write!(f, "Error while loading function: {}", msg)
            },
            Self::UnsupportedDTypeForOp(dtype, op) => {
                write!(f, "Unsupported dtype {} for operation {}", dtype, op)
            },
            Self::FailedToCreateComputeFunction => {
                write!(f, "Failed to create compute function")
            },
            Self::FailedToCreateResource(msg) => {
                write!(f, "Failed to create metal resource: {}", msg)
            },
            Self::FailedToCreatePipeline(msg) => {
                write!(f, "Failed to create pipeline: {}", msg)
            },
            Self::MatMulNonContiguous {
                lhs_stride,
                rhs_stride,
                mnk,
            } => {
                write!(
                    f,
                    "Invalid matmul arguments lhs_stride={:?} rhs_stride={:?} mnk={:?}",
                    lhs_stride, rhs_stride, mnk
                )
            },
            Self::WithBacktrace { inner, backtrace } => {
                write!(f, "{}\n{}", inner, backtrace)
            },
        }
    }
}

impl std::error::Error for MetalKernelError {}

impl MetalKernelError {
    pub fn bt(self) -> Self {
        let backtrace = std::backtrace::Backtrace::capture();
        match backtrace.status() {
            std::backtrace::BacktraceStatus::Disabled | std::backtrace::BacktraceStatus::Unsupported => self,
            _ => Self::WithBacktrace {
                inner: Box::new(self),
                backtrace: Box::new(backtrace),
            },
        }
    }
}

impl<T> From<std::sync::PoisonError<T>> for MetalKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}
