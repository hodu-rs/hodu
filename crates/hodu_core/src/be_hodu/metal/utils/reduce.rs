use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    op::ReduceOp,
    types::{dtype::DType, layout::Layout},
};

pub fn reduce_map(
    storage: &MetalStorage,
    layout: &Layout,
    reduce_op: ReduceOp,
    dims: &[usize],
    keep_dim: bool,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_reduce, utils::BufferOffset};

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

    macro_rules! dispatch_reduce {
        ($kernel_mod:ident, $variant:ident) => {
            match dtype {
                DType::BF16 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("reduce_{:?}", reduce_op),
                    })
                },
            }
        };
    }

    macro_rules! dispatch_reduce_bool_output {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BOOL => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BOOL,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::BF16 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_reduce(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        &reduce_dims,
                        reduce_size,
                        keep_dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("reduce_{:?}", reduce_op),
                    })
                },
            }
        };
    }

    // Match reduce operation and dispatch to appropriate kernel
    match reduce_op {
        ReduceOp::Sum => dispatch_reduce!(reduce_sum, Sum),
        ReduceOp::Max => dispatch_reduce!(reduce_max, Max),
        ReduceOp::Min => dispatch_reduce!(reduce_min, Min),
        ReduceOp::Prod => dispatch_reduce!(reduce_prod, Prod),
        ReduceOp::Mean => {
            // Mean only supports floating point types
            match dtype {
                DType::BF16 | DType::F16 | DType::F32 => {
                    dispatch_reduce!(reduce_mean, Mean);
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
                    dispatch_reduce!(reduce_norm, Norm);
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "reduce_norm".to_string(),
                    })
                },
            }
        },
        ReduceOp::ArgMax => dispatch_reduce!(reduce_argmax, ArgMax),
        ReduceOp::ArgMin => dispatch_reduce!(reduce_argmin, ArgMin),
        ReduceOp::Any => dispatch_reduce_bool_output!(reduce_any),
        ReduceOp::All => dispatch_reduce_bool_output!(reduce_all),
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: format!("reduce_{:?}", reduce_op),
            })
        },
    }

    Ok(MetalStorage::new(output, device.clone(), num_els, output_dtype))
}
