use crate::{
    compat::*,
    tensor::TensorId,
    types::{backend::Backend, device::Device, dtype::DType},
};

pub enum HoduError {
    // Device
    DeviceMismatch {
        expected: Device,
        got: Device,
    },
    DeviceConflictInOp {
        left: Device,
        right: Device,
        op: String,
    },
    UnsupportedDevice(Device),
    UnsupportedBackend(Backend),
    // DType
    DTypeMismatch {
        expected: DType,
        got: DType,
    },
    DTypeConflictInOp {
        left: DType,
        right: DType,
        op: String,
    },
    UnsupportedDType(DType),
    // Shape and Layout
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    IncompatibleShapes {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
        op: String,
    },
    InvalidLayout {
        tensor_id: TensorId,
        reason: String,
    },
    // Builder
    BuilderContextAlreadyActive(String),
    BuilderContextNotActive,
    BuilderAlreadyEnded(String),
    BuilderValidationFailed(String),
    MissingInputs(Vec<String>),
    MissingOutputs(Vec<String>),
    // Static Tensor Creation
    StaticTensorCreationRequiresBuilderContext,
    // Tensor
    TensorNotFound(TensorId),
    StorageNotFound(TensorId),
    StorageCorrupted(TensorId),
    // Gradient
    GradientTapeCorrupted,
    VjpFunctionNotFound(String),
    GradientComputationFailed(String),
    RequiresGradNotSet(TensorId),
    GradientNotComputed(TensorId),
    // Script
    InvalidScriptVersion(String),
    ScriptValidationFailed(String),
    CompressionError(String),
    DecompressionError(String),
    // IO and Serialization
    IoError(String),
    SerializationError(String),
    // Internal
    InternalError(String),
}

impl fmt::Display for HoduError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceMismatch { expected, got } => {
                write!(f, "Device mismatch - expected {expected}, got {got}")
            },
            Self::DeviceConflictInOp { left, right, op } => {
                write!(
                    f,
                    "Device conflict in operation '{op}' - left operand on {left}, right operand on {right}"
                )
            },
            Self::UnsupportedDevice(device) => {
                write!(f, "Unsupported device - {device}")
            },
            Self::UnsupportedBackend(backend) => {
                write!(f, "Unsupported backend - {backend:?}")
            },
            Self::DTypeMismatch { expected, got } => {
                write!(f, "DType mismatch - expected {expected}, got {got}")
            },
            Self::DTypeConflictInOp { left, right, op } => {
                write!(
                    f,
                    "DType conflict in operation '{op}' - left operand is {left}, right operand is {right}"
                )
            },
            Self::UnsupportedDType(dtype) => {
                write!(f, "Unsupported dtype - {dtype}")
            },
            Self::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch - expected {expected:?}, got {got:?}")
            },
            Self::IncompatibleShapes { lhs, rhs, op } => {
                write!(
                    f,
                    "Incompatible shapes in operation '{op}' - lhs: {lhs:?}, rhs: {rhs:?}"
                )
            },
            Self::InvalidLayout { tensor_id, reason } => {
                write!(f, "Invalid layout for tensor {tensor_id:?} - {reason}")
            },
            Self::BuilderContextAlreadyActive(name) => {
                write!(
                    f,
                    "Builder context already active - '{name}'. Call end() first before starting a new context."
                )
            },
            Self::BuilderContextNotActive => {
                write!(f, "No active builder context - Call start() first.")
            },
            Self::BuilderAlreadyEnded(name) => {
                write!(f, "Builder '{name}' has already been ended and cannot be restarted.")
            },
            Self::BuilderValidationFailed(msg) => {
                write!(f, "Builder validation failed - {msg}")
            },
            Self::MissingInputs(inputs) => {
                write!(f, "Missing required inputs - {inputs:?}")
            },
            Self::MissingOutputs(outputs) => {
                write!(f, "Missing required outputs - {outputs:?}")
            },
            Self::StaticTensorCreationRequiresBuilderContext => {
                write!(f, "Static tensor creation (input, constant) requires an active builder context - Call builder.start() first.")
            },
            Self::TensorNotFound(tensor_id) => {
                write!(f, "Tensor not found - TensorId({tensor_id:?})")
            },
            Self::StorageNotFound(tensor_id) => {
                write!(f, "Storage not found for tensor - TensorId({tensor_id:?})")
            },
            Self::StorageCorrupted(tensor_id) => {
                write!(f, "Storage corrupted for tensor - TensorId({tensor_id:?})")
            },
            Self::GradientTapeCorrupted => {
                write!(f, "Gradient computation tape is corrupted")
            },
            Self::VjpFunctionNotFound(op) => {
                write!(f, "VJP function not found for operation - {op}")
            },
            Self::GradientComputationFailed(msg) => {
                write!(f, "Gradient computation failed - {msg}")
            },
            Self::RequiresGradNotSet(tensor_id) => {
                write!(f, "Requires grad not set for tensor - TensorId({tensor_id:?})")
            },
            Self::GradientNotComputed(tensor_id) => {
                write!(
                    f,
                    "Gradient not computed for tensor - TensorId({tensor_id:?}). Call backward() first."
                )
            },
            Self::InvalidScriptVersion(version) => {
                write!(f, "Invalid script version - {version}")
            },
            Self::ScriptValidationFailed(msg) => {
                write!(f, "Script validation failed - {msg}")
            },
            Self::CompressionError(msg) => {
                write!(f, "Compression error - {msg}")
            },
            Self::DecompressionError(msg) => {
                write!(f, "Decompression error - {msg}")
            },
            Self::IoError(msg) => write!(f, "IO error - {msg}"),
            Self::SerializationError(msg) => write!(f, "Serialization error - {msg}"),
            Self::InternalError(msg) => write!(f, "Internal error - {msg}"),
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

pub type HoduResult<T> = Result<T, HoduError>;
