use crate::{
    compat::*,
    ops::Op,
    tensor::TensorId,
    types::{DType, Device, Shape},
};

/// Main error type for hodu_core.
///
/// This enum covers all possible error conditions that can occur
/// during tensor operations.
#[derive(Clone)]
pub enum HoduError {
    // ===== Device Errors =====
    /// Device mismatch between expected and actual device.
    DeviceMismatch { expected: Device, got: Device },
    /// Device conflict in a binary operation.
    DeviceConflictInOp { left: Device, right: Device, op: Op },
    /// Unsupported device type.
    UnsupportedDevice(Device),

    // ===== DType Errors =====
    /// Data type mismatch between expected and actual dtype.
    DTypeMismatch { expected: DType, got: DType },
    /// Data type conflict in a binary operation.
    DTypeConflictInOp { left: DType, right: DType, op: Op },
    /// Unsupported dtype with reason.
    UnsupportedDType { dtype: DType, reason: String },
    /// Unsupported dtype for a specific operation.
    UnsupportedDTypeForOp { dtype: DType, op: Op },
    /// Unsupported dtype for a specific device.
    UnsupportedDTypeForDevice { dtype: DType, device: Device },

    // ===== Shape and Layout Errors =====
    /// Shape mismatch between expected and actual shapes.
    ShapeMismatch { expected: Shape, got: Shape },
    /// Size mismatch between expected and actual sizes.
    SizeMismatch { expected: usize, got: usize },
    /// Incompatible shapes in a binary operation.
    IncompatibleShapes { lhs: Shape, rhs: Shape, op: Op },
    /// Invalid layout configuration.
    InvalidLayout { reason: String },
    /// Invalid axis for the given shape.
    InvalidAxis { axis: i32, ndim: usize },

    // ===== Tensor Errors =====
    /// Tensor not found in the global registry.
    TensorNotFound(TensorId),
    /// Storage not found for a tensor.
    StorageNotFound(TensorId),
    /// Storage data is corrupted or inaccessible.
    StorageCorrupted(TensorId),
    /// Requires grad is not set for tensor (only float tensors can have gradients).
    RequiresGradNotSet(TensorId),

    // ===== Capture Errors =====
    /// Capture board is not active
    CaptureNotActive,
    /// Capture board not found with given ID
    CaptureNotFound(String),

    // ===== Backend Errors =====
    /// Backend operation failed
    BackendError(String),
    /// CPU kernel error
    CpuKernelError(String),
    /// CUDA kernel error
    #[cfg(feature = "cuda")]
    CudaKernelError(String),
    /// Metal kernel error
    #[cfg(feature = "metal")]
    MetalKernelError(String),

    // ===== Gradient Errors =====
    /// VJP (Vector-Jacobian Product) function not found for operation.
    VjpFunctionNotFound(String),
    /// Gradient tape is corrupted or inaccessible.
    GradientTapeCorrupted,
    /// Gradient computation failed.
    GradientComputationFailed(String),
    /// Gradient has not been computed for this tensor.
    GradientNotComputed(TensorId),

    // ===== IO Errors =====
    /// I/O operation failed.
    IoError(String),
    /// File not found.
    FileNotFound(String),

    // ===== Serialization Errors =====
    /// Invalid argument provided.
    InvalidArgument(String),
    /// Serialization failed.
    SerializationFailed(String),
    /// Deserialization failed.
    DeserializationFailed(String),
    /// Unsupported operation.
    UnsupportedOperation(String),

    // ===== Internal Errors =====
    /// Internal error with a descriptive message.
    InternalError(String),
    /// Not yet implemented feature.
    NotImplemented(String),
    /// Unsupported platform.
    UnsupportedPlatform(String),
}

impl fmt::Display for HoduError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Device Errors
            Self::DeviceMismatch { expected, got } => {
                write!(f, "device mismatch: expected {:?}, got {:?}", expected, got)
            },
            Self::DeviceConflictInOp { left, right, op } => {
                write!(
                    f,
                    "device conflict in operation {:?}: left on {:?}, right on {:?}",
                    op, left, right
                )
            },
            Self::UnsupportedDevice(device) => {
                write!(f, "unsupported {:?} device", device)
            },

            // DType Errors
            Self::DTypeMismatch { expected, got } => {
                write!(f, "dtype mismatch: expected {:?}, got {:?}", expected, got)
            },
            Self::DTypeConflictInOp { left, right, op } => {
                write!(
                    f,
                    "dtype conflict in operation {:?}: left is {:?}, right is {:?}",
                    op, left, right
                )
            },
            Self::UnsupportedDType { dtype, reason } => {
                write!(f, "unsupported dtype {:?}: {}", dtype, reason)
            },
            Self::UnsupportedDTypeForOp { dtype, op } => {
                write!(f, "unsupported dtype {:?} for operation {:?}", dtype, op)
            },
            Self::UnsupportedDTypeForDevice { dtype, device } => {
                write!(f, "unsupported dtype {:?} for {:?} device", dtype, device)
            },

            // Shape and Layout Errors
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {:?}, got {:?}", expected, got)
            },
            Self::SizeMismatch { expected, got } => {
                write!(f, "size mismatch: expected {}, got {}", expected, got)
            },
            Self::IncompatibleShapes { lhs, rhs, op } => {
                write!(
                    f,
                    "incompatible shapes in operation {:?}: lhs {:?}, rhs {:?}",
                    op, lhs, rhs
                )
            },
            Self::InvalidLayout { reason } => {
                write!(f, "invalid layout: {}", reason)
            },
            Self::InvalidAxis { axis, ndim } => {
                write!(f, "invalid axis {} for {}-dimensional tensor", axis, ndim)
            },

            // Tensor Errors
            Self::TensorNotFound(id) => {
                write!(f, "tensor not found: id={:?}", id)
            },
            Self::StorageNotFound(id) => {
                write!(f, "storage not found for tensor: id={:?}", id)
            },
            Self::StorageCorrupted(id) => {
                write!(f, "storage corrupted for tensor: id={:?}", id)
            },
            Self::RequiresGradNotSet(id) => {
                write!(f, "requires_grad not set or not allowed for tensor: id={:?}", id)
            },

            // Capture Errors
            Self::CaptureNotActive => write!(f, "capture board is not active"),
            Self::CaptureNotFound(msg) => write!(f, "capture board not found: {}", msg),

            // Backend Errors
            Self::BackendError(msg) => write!(f, "backend error: {}", msg),
            Self::CpuKernelError(msg) => write!(f, "cpu kernel error: {}", msg),
            #[cfg(feature = "cuda")]
            Self::CudaKernelError(msg) => write!(f, "cuda kernel error: {}", msg),
            #[cfg(feature = "metal")]
            Self::MetalKernelError(msg) => write!(f, "metal kernel error: {}", msg),

            // Gradient Errors
            Self::VjpFunctionNotFound(msg) => {
                write!(f, "vjp function not found: {}", msg)
            },
            Self::GradientTapeCorrupted => {
                write!(f, "gradient tape is corrupted or inaccessible")
            },
            Self::GradientComputationFailed(msg) => {
                write!(f, "gradient computation failed: {}", msg)
            },
            Self::GradientNotComputed(id) => {
                write!(f, "gradient not computed for tensor: {}", id)
            },

            // IO Errors
            Self::IoError(msg) => {
                write!(f, "io error: {}", msg)
            },
            Self::FileNotFound(path) => {
                write!(f, "file not found: {}", path)
            },

            // Serialization Errors
            Self::InvalidArgument(msg) => {
                write!(f, "invalid argument: {}", msg)
            },
            Self::SerializationFailed(msg) => {
                write!(f, "serialization failed: {}", msg)
            },
            Self::DeserializationFailed(msg) => {
                write!(f, "deserialization failed: {}", msg)
            },
            Self::UnsupportedOperation(msg) => {
                write!(f, "unsupported operation: {}", msg)
            },

            // Internal Errors
            Self::InternalError(msg) => {
                write!(f, "internal error: {}", msg)
            },
            Self::NotImplemented(feature) => {
                write!(f, "not implemented: {}", feature)
            },
            Self::UnsupportedPlatform(msg) => {
                write!(f, "unsupported platform: {}", msg)
            },
        }
    }
}

impl fmt::Debug for HoduError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HoduError {}

// Conversion from common error types
#[cfg(feature = "std")]
impl From<std::io::Error> for HoduError {
    fn from(e: std::io::Error) -> Self {
        HoduError::IoError(e.to_string())
    }
}

#[cfg(feature = "std")]
impl From<std::string::FromUtf8Error> for HoduError {
    fn from(e: std::string::FromUtf8Error) -> Self {
        HoduError::InternalError(format!("utf-8 conversion error: {}", e))
    }
}

// Conversion from hodu_cpu_kernels error
impl From<hodu_cpu_kernels::CpuKernelError> for HoduError {
    fn from(e: hodu_cpu_kernels::CpuKernelError) -> Self {
        HoduError::CpuKernelError(format!("{:?}", e))
    }
}

// Conversion from hodu_cuda_kernels error
#[cfg(feature = "cuda")]
impl From<hodu_cuda_kernels::error::CudaKernelError> for HoduError {
    fn from(e: hodu_cuda_kernels::error::CudaKernelError) -> Self {
        HoduError::CudaKernelError(format!("{:?}", e))
    }
}

// Conversion from hodu_metal_kernels error
#[cfg(feature = "metal")]
impl From<hodu_metal_kernels::error::MetalKernelError> for HoduError {
    fn from(e: hodu_metal_kernels::error::MetalKernelError) -> Self {
        HoduError::MetalKernelError(format!("{:?}", e))
    }
}

// Conversion from PoisonError (for RwLock/Mutex)
#[cfg(feature = "std")]
impl<T> From<std::sync::PoisonError<T>> for HoduError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        HoduError::InternalError(format!("lock poisoned: {}", e))
    }
}

/// Result type alias for hodu_core operations.
pub type HoduResult<T> = Result<T, HoduError>;
