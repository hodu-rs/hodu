pub use crate::{
    backends::op::{
        window_reduction::WindowReduction, BinaryLogicalOp, BinaryOp, CastOp, CmpOp, CmpScalarOp, ConcatOp, IndexingOp,
        MatrixOp, MemoryOp, Op, ReduceOp, ShapeOp, SplitOp, UnaryLogicalOp, UnaryOp, UnaryScalarOp, WindowingOp,
    },
    compat::*,
    error::{HoduError, HoduResult},
    tensor::Tensor,
    types::{backend::Backend, device::Device, dtype::DType},
};

pub fn validate_same_device(tensors: &[&Tensor], op: &str) -> HoduResult<()> {
    if tensors.is_empty() {
        return Ok(());
    }

    let first_device = tensors[0].get_device();

    for tensor in tensors.iter().skip(1) {
        let current_device = tensor.get_device();
        if current_device != first_device {
            return Err(HoduError::DeviceConflictInOp {
                left: first_device,
                right: current_device,
                op: op.to_string(),
            });
        }
    }

    Ok(())
}

pub fn validate_dtype_for_device(dtype: DType, device: &Device, op: &str) -> HoduResult<()> {
    match device {
        Device::CPU => Ok(()),
        Device::CUDA(_) => Ok(()),
        Device::Metal => {
            // metal: f8e4m3, f8e5m2, f64 not supported
            match dtype {
                DType::F8E4M3 | DType::F8E5M2 | DType::F64 => Err(HoduError::UnsupportedDTypeForDeviceInOp {
                    dtype,
                    device: *device,
                    op: op.to_string(),
                }),
                _ => Ok(()),
            }
        },
    }
}

pub fn validate_dtype_for_backend(dtype: DType, backend: &Backend, op: &str) -> HoduResult<()> {
    match backend {
        Backend::HODU => Ok(()),
        Backend::XLA => {
            // xla: f8e4m3, f8e5m2 not supported
            match dtype {
                DType::F8E4M3 | DType::F8E5M2 => Err(HoduError::UnsupportedDTypeForBackendInOp {
                    dtype,
                    backend: *backend,
                    op: op.to_string(),
                }),
                _ => Ok(()),
            }
        },
    }
}

pub fn validate_dtype_for_op(dtype: DType, op: &Op) -> HoduResult<()> {
    match op {
        // Binary operations
        Op::Binary(op, ..) => match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("{:?}", op).to_lowercase(),
                    });
                }
            },
            BinaryOp::Div | BinaryOp::Pow => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("{:?}", op).to_lowercase(),
                    });
                }
            },
            BinaryOp::Maximum | BinaryOp::Minimum => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("{:?}", op).to_lowercase(),
                    });
                }
            },
        },

        // Binary logical operations - all types supported
        Op::BinaryLogical(..) => {
            // All types supported (-)
        },

        // Comparison operations
        Op::Cmp(op, ..) => match op {
            CmpOp::Eq | CmpOp::Ne => {
                // All types supported (-)
            },
            CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("{:?}", op).to_lowercase(),
                    });
                }
            },
        },

        // Comparison with scalar
        Op::CmpScalar(op, ..) => match op {
            CmpScalarOp::EqScalar | CmpScalarOp::NeScalar => {
                // All types supported (-)
            },
            CmpScalarOp::LtScalar | CmpScalarOp::LeScalar | CmpScalarOp::GtScalar | CmpScalarOp::GeScalar => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("{:?}", op).to_lowercase(),
                    });
                }
            },
        },

        // Unary operations
        Op::Unary(op, ..) => match op {
            UnaryOp::Neg | UnaryOp::Abs | UnaryOp::Sign => {
                if dtype == DType::BOOL || dtype.is_uint() {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("{:?}", op).to_lowercase(),
                    });
                }
            },
            UnaryOp::Square => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "square".to_string(),
                    });
                }
            },
            UnaryOp::Sqrt | UnaryOp::Recip => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("{:?}", op).to_lowercase(),
                    });
                }
            },
            UnaryOp::Relu => {
                if dtype == DType::BOOL || dtype.is_uint() {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "relu".to_string(),
                    });
                }
            },
            UnaryOp::Sigmoid | UnaryOp::Tanh | UnaryOp::Gelu | UnaryOp::Softplus | UnaryOp::Silu | UnaryOp::Mish => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("{:?}", op).to_lowercase(),
                    });
                }
            },
            UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Tan => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("{:?}", op).to_lowercase(),
                    });
                }
            },
            UnaryOp::Exp | UnaryOp::Exp2 | UnaryOp::Exp10 | UnaryOp::Ln | UnaryOp::Log2 | UnaryOp::Log10 => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("{:?}", op).to_lowercase(),
                    });
                }
            },
        },

        // Unary logical operations - all types supported
        Op::UnaryLogical(..) => {
            // All types supported (-)
        },

        // Unary with scalar
        Op::UnaryScalar(op, ..) => match op {
            UnaryScalarOp::AddScalar | UnaryScalarOp::SubScalar | UnaryScalarOp::MulScalar => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("{:?}", op).to_lowercase(),
                    });
                }
            },
            UnaryScalarOp::DivScalar | UnaryScalarOp::PowScalar => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("{:?}", op).to_lowercase(),
                    });
                }
            },
            UnaryScalarOp::MaximumScalar | UnaryScalarOp::MinimumScalar => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("{:?}", op).to_lowercase(),
                    });
                }
            },
            UnaryScalarOp::LeakyRelu | UnaryScalarOp::Elu | UnaryScalarOp::Prelu => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("{:?}", op).to_lowercase(),
                    });
                }
            },
        },

        // Matrix operations
        Op::Matrix(op, ..) => {
            if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: format!("{:?}", op).to_lowercase(),
                });
            }
        },

        // Reduce operations
        Op::Reduce(op, ..) => match op {
            ReduceOp::Sum => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "sum".to_string(),
                    });
                }
            },
            ReduceOp::Mean | ReduceOp::Std | ReduceOp::Var | ReduceOp::Norm => {
                if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: op.to_string(),
                    });
                }
            },
            ReduceOp::Max | ReduceOp::Min => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: op.to_string(),
                    });
                }
            },
            ReduceOp::Prod => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "prod".to_string(),
                    });
                }
            },
            ReduceOp::ArgMax | ReduceOp::ArgMin | ReduceOp::Any | ReduceOp::All => {
                // All types supported (-)
            },
        },

        // Concat and Split - all types supported
        Op::Concat(..) | Op::Split(..) => {
            // All types supported (-)
        },

        // Indexing operations
        Op::Indexing(op, ..) => match op {
            IndexingOp::IndexSelect | IndexingOp::IndexPut | IndexingOp::Gather | IndexingOp::Scatter => {
                // All types supported (-)
            },
            IndexingOp::ScatterAdd | IndexingOp::ScatterMax | IndexingOp::ScatterMin => {
                if dtype == DType::BOOL {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("{:?}", op).to_lowercase(),
                    });
                }
            },
        },

        // Convolution operations
        Op::Conv(op, ..) => {
            if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: format!("{:?}", op).to_lowercase(),
                });
            }
        },

        // Windowing operations
        Op::Windowing(WindowingOp::ReduceWindow, _, params) => {
            if params.is_empty() {
                return Ok(()); // params가 없으면 검증 스킵
            }

            // Extract reduction type from params
            // params structure: [rank, window_shape..., strides..., padding_lo..., padding_hi..., reduction_type]
            let rank = params[0].to_u32() as usize;
            let expected_len = 1 + rank + rank + rank + rank + 1;

            if params.len() == expected_len {
                let reduction_type_val = params[params.len() - 1].to_u32();

                match reduction_type_val {
                    0 | 3 => {
                        // Max (0) and Min (3): only bool not supported
                        if dtype == DType::BOOL {
                            return Err(HoduError::UnsupportedDType {
                                dtype,
                                op: if reduction_type_val == 0 {
                                    "reduce_window_max"
                                } else {
                                    "reduce_window_min"
                                }
                                .to_string(),
                            });
                        }
                    },
                    1 => {
                        // Mean (1): bool, U, I not supported
                        if dtype == DType::BOOL || dtype.is_uint() || dtype.is_int() {
                            return Err(HoduError::UnsupportedDType {
                                dtype,
                                op: "reduce_window_mean".to_string(),
                            });
                        }
                    },
                    2 => {
                        // Sum (2): only bool not supported
                        if dtype == DType::BOOL {
                            return Err(HoduError::UnsupportedDType {
                                dtype,
                                op: "reduce_window_sum".to_string(),
                            });
                        }
                    },
                    _ => {
                        // Unknown reduction type, skip validation
                    },
                }
            }
        },

        // Shape, Shape with Scalars, Cast, Memory operations - all types supported
        Op::Shape(..) | Op::ShapeScalars(..) | Op::Cast(..) | Op::Memory(..) => {
            // All types supported (-)
        },

        Op::Dummy => {
            // No validation needed
        },
    }

    Ok(())
}

pub fn validate_same_dtype(tensors: &[&Tensor], op: &str) -> HoduResult<()> {
    if tensors.is_empty() {
        return Ok(());
    }

    let first_dtype = tensors[0].get_dtype();

    for tensor in tensors.iter().skip(1) {
        let current_dtype = tensor.get_dtype();
        if current_dtype != first_dtype {
            return Err(HoduError::DTypeConflictInOp {
                left: first_dtype,
                right: current_dtype,
                op: op.to_string(),
            });
        }
    }

    Ok(())
}

pub fn validate_indices_dtype(indices: &Tensor, op: &str) -> HoduResult<()> {
    let dtype = indices.get_dtype();
    if !dtype.is_int() && !dtype.is_uint() {
        return Err(HoduError::UnsupportedDType {
            dtype,
            op: format!("{} - indices must be integer type", op),
        });
    }
    Ok(())
}
