use crate::{
    error::{HoduError, HoduResult},
    ops::{BinaryOp, CmpOp, CmpScalarOp, IndexingOp, Op, ReduceOp, UnaryOp, UnaryScalarOp, WindowingOp},
    tensor::Tensor,
    types::{DType, Device},
};

pub fn validate_dtype_for_device(dtype: DType, device: Device) -> HoduResult<()> {
    match device {
        Device::CPU => Ok(()),
        #[cfg(feature = "cuda")]
        Device::CUDA(_) => Ok(()),
        #[cfg(feature = "metal")]
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

pub fn validate_dtype_for_op(dtype: DType, op: Op) -> HoduResult<()> {
    match op {
        // Binary operations
        Op::Binary(inner_op) => match inner_op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => {
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
            UnaryOp::Sigmoid | UnaryOp::Tanh | UnaryOp::Gelu | UnaryOp::Softplus | UnaryOp::Silu | UnaryOp::Mish => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Tan => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
                }
            },
            UnaryOp::Exp | UnaryOp::Exp2 | UnaryOp::Exp10 | UnaryOp::Ln | UnaryOp::Log2 | UnaryOp::Log10 => {
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
            UnaryScalarOp::AddScalar | UnaryScalarOp::SubScalar | UnaryScalarOp::MulScalar => {
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

        // Shape, ShapeScalars, Cast, Memory operations - all types supported
        Op::Shape(_) | Op::ShapeScalars(_) | Op::Cast(_) | Op::Memory(_) | Op::Dummy => {
            // All types supported
        },
    }

    Ok(())
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
    let dtype = indices.dtype();
    if !dtype.is_int() && !dtype.is_uint() {
        return Err(HoduError::UnsupportedDTypeForOp { dtype, op });
    }
    Ok(())
}
