use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    op::ReduceOp,
    types::{dtype::DType, layout::Layout},
};

/// Common dtype dispatch macro for reduce operations.
macro_rules! dispatch_reduce_dtype {
    ($kernel_mod:ident, $dtype:expr, $device:expr, $command_buffer:expr, $kernels:expr, $shape:expr, $input:expr, $strides:expr, $offset:expr, $reduce_dims:expr, $reduce_size:expr, $keep_dim:expr, $output:expr, $reduce_op:expr) => {
        match $dtype {
            DType::BF16 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::BF16,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::F16 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::F16,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::F32 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::F32,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            #[cfg(feature = "u8")]
            DType::U8 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::U8,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::U16 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::U16,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            #[cfg(feature = "u32")]
            DType::U32 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::U32,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            #[cfg(feature = "u64")]
            DType::U64 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::U64,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::I8 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::I8,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            #[cfg(feature = "i16")]
            DType::I16 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::I16,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::I32 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::I32,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            #[cfg(feature = "i64")]
            DType::I64 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::I64,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype: $dtype,
                    op: format!("reduce_{:?}", $reduce_op),
                })
            },
        }
    };
}

/// Dispatch macro for reduce operations with bool output (any/all).
macro_rules! dispatch_reduce_bool_dtype {
    ($kernel_mod:ident, $dtype:expr, $device:expr, $command_buffer:expr, $kernels:expr, $shape:expr, $input:expr, $strides:expr, $offset:expr, $reduce_dims:expr, $reduce_size:expr, $keep_dim:expr, $output:expr, $reduce_op:expr) => {
        match $dtype {
            DType::BOOL => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::BOOL,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::BF16 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::BF16,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::F16 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::F16,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::F32 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::F32,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            #[cfg(feature = "u8")]
            DType::U8 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::U8,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::U16 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::U16,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            #[cfg(feature = "u32")]
            DType::U32 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::U32,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            #[cfg(feature = "u64")]
            DType::U64 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::U64,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::I8 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::I8,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            #[cfg(feature = "i16")]
            DType::I16 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::I16,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::I32 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::I32,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            #[cfg(feature = "i64")]
            DType::I64 => {
                hodu_metal_kernels::kernels::call_reduce(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$kernel_mod::I64,
                    $shape,
                    $input,
                    $strides,
                    $offset,
                    $reduce_dims,
                    $reduce_size,
                    $keep_dim,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype: $dtype,
                    op: format!("reduce_{:?}", $reduce_op),
                })
            },
        }
    };
}

pub fn reduce_map(
    storage: &MetalStorage,
    layout: &Layout,
    reduce_op: ReduceOp,
    dims: &[usize],
    keep_dim: bool,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::utils::BufferOffset;

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    // Handle empty dims (reduce all)
    let reduce_dims: Vec<usize> = if dims.is_empty() {
        (0..ndim).collect()
    } else {
        dims.to_vec()
    };

    // Calculate output shape and reduce_size
    let mut output_shape = shape.to_vec();
    let mut reduce_size = 1usize;
    for &dim in &reduce_dims {
        reduce_size *= shape[dim];
        if keep_dim {
            output_shape[dim] = 1;
        } else {
            output_shape[dim] = 0; // Mark for removal
        }
    }

    if !keep_dim {
        output_shape.retain(|&size| size != 0);
        if output_shape.is_empty() {
            output_shape = vec![1]; // Scalar result
        }
    }

    let num_els: usize = output_shape.iter().product();

    // Determine output dtype based on operation
    let output_dtype = match reduce_op {
        ReduceOp::ArgMax | ReduceOp::ArgMin => DType::I32,
        ReduceOp::Any | ReduceOp::All => DType::BOOL,
        _ => dtype,
    };

    let output = device.new_buffer(num_els, output_dtype, &format!("reduce_{:?}", reduce_op))?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    // Match reduce operation and dispatch to appropriate kernel
    match reduce_op {
        ReduceOp::Sum => dispatch_reduce_dtype!(
            reduce_sum,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            shape,
            input,
            strides,
            offset,
            &reduce_dims,
            reduce_size,
            keep_dim,
            &output,
            reduce_op
        ),
        ReduceOp::Max => dispatch_reduce_dtype!(
            reduce_max,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            shape,
            input,
            strides,
            offset,
            &reduce_dims,
            reduce_size,
            keep_dim,
            &output,
            reduce_op
        ),
        ReduceOp::Min => dispatch_reduce_dtype!(
            reduce_min,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            shape,
            input,
            strides,
            offset,
            &reduce_dims,
            reduce_size,
            keep_dim,
            &output,
            reduce_op
        ),
        ReduceOp::Prod => dispatch_reduce_dtype!(
            reduce_prod,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            shape,
            input,
            strides,
            offset,
            &reduce_dims,
            reduce_size,
            keep_dim,
            &output,
            reduce_op
        ),
        ReduceOp::Mean => {
            // Mean only supports floating point types
            match dtype {
                DType::BF16 | DType::F16 | DType::F32 => {
                    dispatch_reduce_dtype!(
                        reduce_mean,
                        dtype,
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                        reduce_op
                    );
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "reduce_mean".to_string(),
                    })
                },
            }
        },
        ReduceOp::Norm => {
            // Norm only supports floating point types
            match dtype {
                DType::BF16 | DType::F16 | DType::F32 => {
                    dispatch_reduce_dtype!(
                        reduce_norm,
                        dtype,
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                        reduce_op
                    );
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "reduce_norm".to_string(),
                    })
                },
            }
        },
        ReduceOp::ArgMax => dispatch_reduce_dtype!(
            reduce_argmax,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            shape,
            input,
            strides,
            offset,
            &reduce_dims,
            reduce_size,
            keep_dim,
            &output,
            reduce_op
        ),
        ReduceOp::ArgMin => dispatch_reduce_dtype!(
            reduce_argmin,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            shape,
            input,
            strides,
            offset,
            &reduce_dims,
            reduce_size,
            keep_dim,
            &output,
            reduce_op
        ),
        ReduceOp::Any => dispatch_reduce_bool_dtype!(
            reduce_any,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            shape,
            input,
            strides,
            offset,
            &reduce_dims,
            reduce_size,
            keep_dim,
            &output,
            reduce_op
        ),
        ReduceOp::All => dispatch_reduce_bool_dtype!(
            reduce_all,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            shape,
            input,
            strides,
            offset,
            &reduce_dims,
            reduce_size,
            keep_dim,
            &output,
            reduce_op
        ),
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: format!("reduce_{:?}", reduce_op),
            })
        },
    }

    Ok(MetalStorage::new(output, device.clone(), num_els, output_dtype))
}
