use crate::{
    layer::compat::*,
    ops::Op,
    tensor::TensorId,
    types::{Compiler, DType, Device, Shape},
};

/// Main error type for hodu_core.
///
/// This enum covers all possible error conditions that can occur
/// during tensor operations, compilation, and execution.
#[derive(Clone)]
pub enum HoduError {
    // ===== Device Errors =====
    /// Device mismatch between expected and actual device.
    DeviceMismatch { expected: Device, got: Device },
    /// Device conflict in a binary operation.
    DeviceConflictInOp { left: Device, right: Device, op: Op },
    /// Unsupported device type.
    UnsupportedDevice(Device),
    /// Unsupported device type for a specific compiler
    UnsupportedDeviceForCompiler(Device, Compiler),

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
    /// Unsupported dtype for a specific compiler.
    UnsupportedDTypeForCompiler { dtype: DType, compiler: Compiler },

    // ===== Shape and Layout Errors =====
    /// Shape mismatch between expected and actual shapes.
    ShapeMismatch { expected: Shape, got: Shape },
    /// Incompatible shapes in a binary operation.
    IncompatibleShapes { lhs: Shape, rhs: Shape, op: Op },
    /// Invalid layout configuration.
    InvalidLayout { reason: String },
    /// Invalid axis for the given shape.
    InvalidAxis { axis: i32, ndim: u32 },

    // ===== Tensor Errors =====
    /// Tensor not found in the global registry.
    TensorNotFound(TensorId),
    /// Storage not found for a tensor.
    StorageNotFound(TensorId),
    /// Storage data is corrupted or inaccessible.
    StorageCorrupted(TensorId),
    /// Requires grad is not set for tensor (only float tensors can have gradients).
    RequiresGradNotSet(TensorId),

    // ===== IO Errors =====
    /// I/O operation failed.
    IoError(String),
    /// File not found.
    FileNotFound(String),

    // ===== Internal Errors =====
    /// Internal error with a descriptive message.
    InternalError(String),
    /// Not yet implemented feature.
    NotImplemented(String),
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
            Self::UnsupportedDeviceForCompiler(device, compiler) => {
                write!(f, "unsupported {:?} device for {:?} compiler", device, compiler)
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
            Self::UnsupportedDTypeForCompiler { dtype, compiler } => {
                write!(f, "unsupported dtype {:?} for {:?} compiler", dtype, compiler)
            },

            // Shape and Layout Errors
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {:?}, got {:?}", expected, got)
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

            // IO Errors
            Self::IoError(msg) => {
                write!(f, "io error: {}", msg)
            },
            Self::FileNotFound(path) => {
                write!(f, "file not found: {}", path)
            },

            // Internal Errors
            Self::InternalError(msg) => {
                write!(f, "internal error: {}", msg)
            },
            Self::NotImplemented(feature) => {
                write!(f, "not implemented: {}", feature)
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
        HoduError::InternalError(format!("cpu kernel error: {}", e))
    }
}

/// Result type alias for hodu_core_v2 operations.
pub type HoduResult<T> = Result<T, HoduError>;
