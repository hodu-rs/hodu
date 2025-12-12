use crate::{
    error::{HoduError, HoduResult},
    ops::{BinaryOp, CmpOp, CmpScalarOp, IndexingOp, Op, ReduceOp, UnaryOp, UnaryScalarOp, WindowingOp},
    tensor::Tensor,
    types::{DType, Device},
};

#[allow(unused_variables)]
pub fn validate_dtype_for_device(dtype: DType, device: Device) -> HoduResult<()> {
    match device {
        Device::CPU => Ok(()),
        #[cfg(feature = "cuda")]
        Device::CUDA(_) => Ok(()),
        #[cfg(any(feature = "metal", feature = "metal-device"))]
        Device::Metal => {
            // metal: f8e4m3, f8e5m2, f64 not supported
            match dtype {
                DType::F8E4M3 => Err(HoduError::UnsupportedDTypeForDevice { dtype, device }),
                #[cfg(feature = "f8e5m2")]
                DType::F8E5M2 => Err(HoduError::UnsupportedDTypeForDevice { dtype, device }),
                #[cfg(feature = "f64")]
                DType::F64 => Err(HoduError::UnsupportedDTypeForDevice { dtype, device }),
                _ => Ok(()),
            }
        },
    }
}

pub fn validate_same_device(tensors: &[&Tensor], op: Op) -> HoduResult<()> {
    if tensors.is_empty() {
        return Ok(());
    }

    let first_device = tensors[0].device();

    for tensor in tensors.iter().skip(1) {
        let current_device = tensor.device();
        if current_device != first_device {
            return Err(HoduError::DeviceConflictInOp {
                left: first_device,
                right: current_device,
                op,
            });
        }
    }

    Ok(())
}

pub fn validate_same_dtype(tensors: &[&Tensor], op: Op) -> HoduResult<()> {
    if tensors.is_empty() {
        return Ok(());
    }

    // Skip validation if any tensor is a builder input (no storage)
    for tensor in tensors.iter() {
        if !tensor.has_storage() {
            return Ok(());
        }
    }

    let first_dtype = tensors[0].dtype();

    for tensor in tensors.iter().skip(1) {
        let current_dtype = tensor.dtype();
        if current_dtype != first_dtype {
            return Err(HoduError::DTypeConflictInOp {
                left: first_dtype,
                right: current_dtype,
                op,
            });
        }
    }

    Ok(())
}

pub fn validate_indices_dtype(indices: &Tensor, op: Op) -> HoduResult<()> {
    // Skip validation for builder input tensors (no storage)
    if !indices.has_storage() {
        return Ok(());
    }

    let dtype = indices.dtype();
    if !dtype.is_int() && !dtype.is_uint() {
        return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
    }
    Ok(())
}

pub fn validate_dtype_for_op(dtype: DType, op: Op) -> HoduResult<()> {
    match op {
        // Binary operations
        Op::Binary(inner_op) => match inner_op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Rem => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            BinaryOp::Div | BinaryOp::Pow => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            BinaryOp::Maximum | BinaryOp::Minimum => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
        },

        // Binary logical operations - all types supported
        Op::BinaryLogical(_) => {
            // All types supported
        },

        // Comparison operations
        Op::Cmp(inner_op) => match inner_op {
            CmpOp::Eq | CmpOp::Ne => {
                // All types supported
            },
            CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
        },

        // Comparison with scalar
        Op::CmpScalar(inner_op) => match inner_op {
            CmpScalarOp::EqScalar | CmpScalarOp::NeScalar => {
                // All types supported
            },
            CmpScalarOp::LtScalar | CmpScalarOp::LeScalar | CmpScalarOp::GtScalar | CmpScalarOp::GeScalar => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
        },

        // Unary operations
        Op::Unary(inner_op) => match inner_op {
            UnaryOp::Neg | UnaryOp::Abs | UnaryOp::Sign => {
                if dtype == DType::BOOL || dtype.is_uint() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            UnaryOp::Square => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            UnaryOp::Sqrt | UnaryOp::Recip => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            UnaryOp::Relu => {
                if dtype == DType::BOOL || dtype.is_uint() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            UnaryOp::Sigmoid
            | UnaryOp::HardSigmoid
            | UnaryOp::Gelu
            | UnaryOp::Softplus
            | UnaryOp::Silu
            | UnaryOp::HardSilu
            | UnaryOp::Mish => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Tan | UnaryOp::Asin | UnaryOp::Acos | UnaryOp::Atan => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            UnaryOp::Sinh | UnaryOp::Cosh | UnaryOp::Tanh | UnaryOp::Asinh | UnaryOp::Acosh | UnaryOp::Atanh => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            UnaryOp::Erf => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            UnaryOp::Exp | UnaryOp::Exp2 | UnaryOp::Exp10 | UnaryOp::Ln | UnaryOp::Log2 | UnaryOp::Log10 => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            UnaryOp::Ceil | UnaryOp::Floor | UnaryOp::Round => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
        },

        // Unary logical operations - all types supported
        Op::UnaryLogical(_) => {
            // All types supported
        },

        // Unary with scalar
        Op::UnaryScalar(inner_op) => match inner_op {
            UnaryScalarOp::AddScalar
            | UnaryScalarOp::SubScalar
            | UnaryScalarOp::MulScalar
            | UnaryScalarOp::RemScalar => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            UnaryScalarOp::DivScalar | UnaryScalarOp::PowScalar => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            UnaryScalarOp::MaximumScalar | UnaryScalarOp::MinimumScalar => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            UnaryScalarOp::LeakyRelu | UnaryScalarOp::Elu | UnaryScalarOp::Prelu => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
        },

        // Matrix operations
        Op::Matrix(_) => {
            if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
            }
        },

        // Reduce operations
        Op::Reduce(inner_op) => match inner_op {
            ReduceOp::Sum | ReduceOp::Prod => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            ReduceOp::Mean | ReduceOp::Std | ReduceOp::Var | ReduceOp::Norm => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            ReduceOp::Max | ReduceOp::Min => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            ReduceOp::ArgMax | ReduceOp::ArgMin | ReduceOp::Any | ReduceOp::All => {
                // All types supported
            },
        },

        // Concat and Split - all types supported
        Op::Concat(_) | Op::Split(_) => {
            // All types supported
        },

        // Indexing operations
        Op::Indexing(inner_op) => match inner_op {
            IndexingOp::IndexSelect | IndexingOp::IndexPut | IndexingOp::Gather | IndexingOp::Scatter => {
                // All types supported
            },
            IndexingOp::ScatterAdd | IndexingOp::ScatterMax | IndexingOp::ScatterMin => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            IndexingOp::Onehot => {
                // Onehot input must be i32 (indices)
                if dtype != DType::I32 {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            IndexingOp::Nonzero => {
                // All types supported for nonzero
            },
        },

        // Convolution operations
        Op::Conv(_) => {
            if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
            }
        },

        // Windowing operations
        Op::Windowing(inner_op) => match inner_op {
            WindowingOp::ReduceWindowMax | WindowingOp::ReduceWindowMin | WindowingOp::ReduceWindowSum => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            WindowingOp::ReduceWindowMean => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
        },

        // Padding operations - all types supported
        Op::Padding(_) => {
            // All types supported
        },

        // Scan operations - numeric types only (no bool)
        Op::Scan(_) => {
            if dtype == DType::BOOL {
                return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
            }
        },

        // Sort operations - numeric types only (no bool)
        Op::Sort(_) => {
            if dtype == DType::BOOL {
                return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
            }
        },

        // Einsum operations - numeric types only (no bool)
        Op::Einsum(_) => {
            if dtype == DType::BOOL {
                return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
            }
        },

        // Resize operations - float types only
        Op::Resize(_) => {
            if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
            }
        },

        // Shape, ShapeScalars, ShapeMemory, Cast, Memory operations - all types supported
        Op::Shape(_) | Op::ShapeScalars(_) | Op::ShapeMemory(_) | Op::Cast(_) | Op::Memory(_) | Op::Dummy => {
            // All types supported
        },
    }

    Ok(())
}

pub fn validate_requires_grad_for_op(op: Op) -> bool {
    match op {
        // Binary operations
        Op::Binary(_) => true,

        // Binary logical operations - no backprop
        Op::BinaryLogical(_) => false, // !

        // Comparison operations - no backprop
        Op::Cmp(_) => false, // !

        // Comparison with scalar - no backprop
        Op::CmpScalar(_) => false, // !

        // Unary operations
        Op::Unary(inner_op) => match inner_op {
            UnaryOp::Neg | UnaryOp::Square | UnaryOp::Sqrt | UnaryOp::Recip => true,
            UnaryOp::Abs | UnaryOp::Sign => false, // !
            UnaryOp::Relu
            | UnaryOp::Sigmoid
            | UnaryOp::HardSigmoid
            | UnaryOp::Gelu
            | UnaryOp::Softplus
            | UnaryOp::Silu
            | UnaryOp::HardSilu
            | UnaryOp::Mish => true,
            UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Tan | UnaryOp::Asin | UnaryOp::Acos | UnaryOp::Atan => true,
            UnaryOp::Sinh | UnaryOp::Cosh | UnaryOp::Tanh | UnaryOp::Asinh | UnaryOp::Acosh | UnaryOp::Atanh => true,
            UnaryOp::Exp | UnaryOp::Exp2 | UnaryOp::Exp10 | UnaryOp::Ln | UnaryOp::Log2 | UnaryOp::Log10 => true,
            UnaryOp::Ceil | UnaryOp::Floor | UnaryOp::Round => false,
            UnaryOp::Erf => true,
        },

        // Unary logical operations - no backprop
        Op::UnaryLogical(_) => false, // !

        // Unary with scalar
        Op::UnaryScalar(_) => true,

        // Matrix operations
        Op::Matrix(_) => true,

        // Reduce operations
        Op::Reduce(inner_op) => match inner_op {
            ReduceOp::Sum | ReduceOp::Mean | ReduceOp::Prod | ReduceOp::Std | ReduceOp::Var | ReduceOp::Norm => true,
            ReduceOp::Max | ReduceOp::Min => true,
            ReduceOp::ArgMax | ReduceOp::ArgMin => false, // !
            ReduceOp::Any | ReduceOp::All => false,       // !
        },

        // Concat and Split
        Op::Concat(_) | Op::Split(_) => true,

        // Indexing operations
        Op::Indexing(_) => true,

        // Convolution operations
        Op::Conv(_) => true,

        // Windowing operations
        Op::Windowing(_) => true,

        // Padding operations
        Op::Padding(_) => true,

        // Scan operations
        Op::Scan(_) => true,

        // Sort operations - topk doesn't support backprop (indices are discrete)
        Op::Sort(_) => false,

        // Einsum operations
        Op::Einsum(_) => true,

        // Resize operations (linear/cubic support backprop, nearest doesn't but we handle at runtime)
        Op::Resize(_) => true,

        // Shape operations
        Op::Shape(_) | Op::ShapeScalars(_) | Op::ShapeMemory(_) => true,

        // Cast operations - no backprop
        Op::Cast(_) => false, // !

        // Memory operations - no backprop
        Op::Memory(_) => false, // !

        Op::Dummy => false,
    }
}
