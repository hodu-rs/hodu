use crate::{
    backends::{
        be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
        op::conv::{
            ParamsConv1D, ParamsConv2D, ParamsConv3D, ParamsConvTranspose1D, ParamsConvTranspose2D,
            ParamsConvTranspose3D,
        },
    },
    error::{HoduError, HoduResult},
    scalar::Scalar,
    types::{dtype::DType, layout::Layout},
};

pub fn binary_map(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    kernel_name: &str,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::utils::BufferOffset;

    let dtype = lhs_storage.get_dtype();
    let device = lhs_storage.get_hodu_device();
    let lhs_shape = lhs_layout.get_shape();
    let lhs_strides = lhs_layout.get_strides();
    let lhs_offset = lhs_layout.get_offset();
    let rhs_shape = rhs_layout.get_shape();
    let rhs_strides = rhs_layout.get_strides();
    let rhs_offset = rhs_layout.get_offset();
    let lhs_ndim = lhs_shape.len();
    let rhs_ndim = rhs_shape.len();

    // Calculate output shape with broadcasting
    let ndim = lhs_ndim.max(rhs_ndim);
    let mut output_shape = vec![1; ndim];

    for i in 0..ndim {
        let lhs_dim = if i < lhs_ndim { lhs_shape[lhs_ndim - 1 - i] } else { 1 };
        let rhs_dim = if i < rhs_ndim { rhs_shape[rhs_ndim - 1 - i] } else { 1 };

        if lhs_dim != 1 && rhs_dim != 1 && lhs_dim != rhs_dim {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
                op: format!("binary operation '{}' - incompatible dimensions", kernel_name),
            });
        }
        output_shape[ndim - 1 - i] = lhs_dim.max(rhs_dim);
    }

    let output_size: usize = output_shape.iter().product();
    let output = device.new_buffer(output_size, dtype, kernel_name)?;
    let command_buffer = device.command_buffer()?;

    let lhs = BufferOffset {
        buffer: lhs_storage.buffer(),
        offset_in_bytes: lhs_offset * dtype.get_size_in_bytes(),
    };
    let rhs = BufferOffset {
        buffer: rhs_storage.buffer(),
        offset_in_bytes: rhs_offset * dtype.get_size_in_bytes(),
    };

    macro_rules! dispatch_dtype {
        ($op:ident) => {
            match dtype {
                DType::BF16 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::BF16,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::F16,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::F32,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U8,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U16,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U32,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U64,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I8,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I16,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I32,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I64,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("binary operation '{}'", kernel_name),
                    });
                },
            }
        };
    }

    match kernel_name {
        "add" => dispatch_dtype!(add),
        "sub" => dispatch_dtype!(sub),
        "mul" => dispatch_dtype!(mul),
        "div" => dispatch_dtype!(div),
        "pow" => dispatch_dtype!(pow),
        "maximum" => dispatch_dtype!(maximum),
        "minimum" => dispatch_dtype!(minimum),
        "eq" => dispatch_dtype!(eq),
        "ne" => dispatch_dtype!(ne),
        "lt" => dispatch_dtype!(lt),
        "le" => dispatch_dtype!(le),
        "gt" => dispatch_dtype!(gt),
        "ge" => dispatch_dtype!(ge),
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: format!("unknown binary operation '{}'", kernel_name),
            });
        },
    }

    Ok(MetalStorage::new(output, device.clone(), output_size, dtype))
}

pub fn binary_logical_map(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    kernel_name: &str,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::utils::BufferOffset;

    let dtype = lhs_storage.get_dtype();
    let device = lhs_storage.get_hodu_device();
    let lhs_shape = lhs_layout.get_shape();
    let lhs_strides = lhs_layout.get_strides();
    let lhs_offset = lhs_layout.get_offset();
    let rhs_shape = rhs_layout.get_shape();
    let rhs_strides = rhs_layout.get_strides();
    let rhs_offset = rhs_layout.get_offset();
    let lhs_ndim = lhs_shape.len();
    let rhs_ndim = rhs_shape.len();

    // Calculate output shape with broadcasting
    let ndim = lhs_ndim.max(rhs_ndim);
    let mut output_shape = vec![1; ndim];

    for i in 0..ndim {
        let lhs_dim = if i < lhs_ndim { lhs_shape[lhs_ndim - 1 - i] } else { 1 };
        let rhs_dim = if i < rhs_ndim { rhs_shape[rhs_ndim - 1 - i] } else { 1 };

        if lhs_dim != 1 && rhs_dim != 1 && lhs_dim != rhs_dim {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
                op: format!("binary logical operation '{}' - incompatible dimensions", kernel_name),
            });
        }
        output_shape[ndim - 1 - i] = lhs_dim.max(rhs_dim);
    }

    let output_size: usize = output_shape.iter().product();
    // Output is always BOOL for logical operations
    let output_dtype = DType::BOOL;
    let output = device.new_buffer(output_size, output_dtype, kernel_name)?;
    let command_buffer = device.command_buffer()?;

    let lhs = BufferOffset {
        buffer: lhs_storage.buffer(),
        offset_in_bytes: lhs_offset * dtype.get_size_in_bytes(),
    };
    let rhs = BufferOffset {
        buffer: rhs_storage.buffer(),
        offset_in_bytes: rhs_offset * dtype.get_size_in_bytes(),
    };

    macro_rules! dispatch_dtype {
        ($op:ident) => {
            match dtype {
                DType::BF16 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::BF16,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::F16,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::F32,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U8,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U16,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U32,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U64,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I8,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I16,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I32,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    hodu_metal_kernels::kernels::call_binary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I64,
                        &output_shape,
                        lhs,
                        lhs_strides,
                        lhs_offset,
                        rhs,
                        rhs_strides,
                        rhs_offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("binary logical operation '{}'", kernel_name),
                    });
                },
            }
        };
    }

    match kernel_name {
        "logical_and" => dispatch_dtype!(logical_and),
        "logical_or" => dispatch_dtype!(logical_or),
        "logical_xor" => dispatch_dtype!(logical_xor),
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: format!("unknown binary logical operation '{}'", kernel_name),
            });
        },
    }

    Ok(MetalStorage::new(output, device.clone(), output_size, output_dtype))
}

pub fn cmp_map(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    kernel_name: &str,
) -> HoduResult<MetalStorage> {
    // cmp operations return bool, so we need special handling
    // For now, call binary_map which handles eq, ne, lt, le, gt, ge with bool output
    use hodu_metal_kernels::utils::BufferOffset;

    let dtype = lhs_storage.get_dtype();
    let device = lhs_storage.get_hodu_device();
    let lhs_shape = lhs_layout.get_shape();
    let lhs_strides = lhs_layout.get_strides();
    let lhs_offset = lhs_layout.get_offset();
    let rhs_shape = rhs_layout.get_shape();
    let rhs_strides = rhs_layout.get_strides();
    let rhs_offset = rhs_layout.get_offset();
    let lhs_ndim = lhs_shape.len();
    let rhs_ndim = rhs_shape.len();

    // Calculate output shape with broadcasting
    let ndim = lhs_ndim.max(rhs_ndim);
    let mut output_shape = vec![1; ndim];

    for i in 0..ndim {
        let lhs_dim = if i < lhs_ndim { lhs_shape[lhs_ndim - 1 - i] } else { 1 };
        let rhs_dim = if i < rhs_ndim { rhs_shape[rhs_ndim - 1 - i] } else { 1 };

        if lhs_dim != 1 && rhs_dim != 1 && lhs_dim != rhs_dim {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
                op: format!("comparison '{}' - incompatible dimensions", kernel_name),
            });
        }
        output_shape[ndim - 1 - i] = lhs_dim.max(rhs_dim);
    }

    let output_size: usize = output_shape.iter().product();
    let output_dtype = DType::BOOL;
    let output = device.new_buffer(output_size, output_dtype, kernel_name)?;
    let command_buffer = device.command_buffer()?;

    let lhs = BufferOffset {
        buffer: lhs_storage.buffer(),
        offset_in_bytes: lhs_offset * dtype.get_size_in_bytes(),
    };
    let rhs = BufferOffset {
        buffer: rhs_storage.buffer(),
        offset_in_bytes: rhs_offset * dtype.get_size_in_bytes(),
    };

    // The binary kernels for cmp already output bool
    macro_rules! dispatch_cmp {
        ($op:ident) => {
            match dtype {
                DType::BF16 => hodu_metal_kernels::kernels::call_binary(
                    device.device(),
                    &command_buffer,
                    device.kernels(),
                    hodu_metal_kernels::kernels::$op::BF16,
                    &output_shape,
                    lhs,
                    lhs_strides,
                    lhs_offset,
                    rhs,
                    rhs_strides,
                    rhs_offset,
                    &output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?,
                DType::F16 => hodu_metal_kernels::kernels::call_binary(
                    device.device(),
                    &command_buffer,
                    device.kernels(),
                    hodu_metal_kernels::kernels::$op::F16,
                    &output_shape,
                    lhs,
                    lhs_strides,
                    lhs_offset,
                    rhs,
                    rhs_strides,
                    rhs_offset,
                    &output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?,
                DType::F32 => hodu_metal_kernels::kernels::call_binary(
                    device.device(),
                    &command_buffer,
                    device.kernels(),
                    hodu_metal_kernels::kernels::$op::F32,
                    &output_shape,
                    lhs,
                    lhs_strides,
                    lhs_offset,
                    rhs,
                    rhs_strides,
                    rhs_offset,
                    &output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?,
                DType::U8 => hodu_metal_kernels::kernels::call_binary(
                    device.device(),
                    &command_buffer,
                    device.kernels(),
                    hodu_metal_kernels::kernels::$op::U8,
                    &output_shape,
                    lhs,
                    lhs_strides,
                    lhs_offset,
                    rhs,
                    rhs_strides,
                    rhs_offset,
                    &output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?,
                DType::U16 => hodu_metal_kernels::kernels::call_binary(
                    device.device(),
                    &command_buffer,
                    device.kernels(),
                    hodu_metal_kernels::kernels::$op::U16,
                    &output_shape,
                    lhs,
                    lhs_strides,
                    lhs_offset,
                    rhs,
                    rhs_strides,
                    rhs_offset,
                    &output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?,
                DType::U32 => hodu_metal_kernels::kernels::call_binary(
                    device.device(),
                    &command_buffer,
                    device.kernels(),
                    hodu_metal_kernels::kernels::$op::U32,
                    &output_shape,
                    lhs,
                    lhs_strides,
                    lhs_offset,
                    rhs,
                    rhs_strides,
                    rhs_offset,
                    &output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?,
                DType::U64 => hodu_metal_kernels::kernels::call_binary(
                    device.device(),
                    &command_buffer,
                    device.kernels(),
                    hodu_metal_kernels::kernels::$op::U64,
                    &output_shape,
                    lhs,
                    lhs_strides,
                    lhs_offset,
                    rhs,
                    rhs_strides,
                    rhs_offset,
                    &output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?,
                DType::I8 => hodu_metal_kernels::kernels::call_binary(
                    device.device(),
                    &command_buffer,
                    device.kernels(),
                    hodu_metal_kernels::kernels::$op::I8,
                    &output_shape,
                    lhs,
                    lhs_strides,
                    lhs_offset,
                    rhs,
                    rhs_strides,
                    rhs_offset,
                    &output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?,
                DType::I16 => hodu_metal_kernels::kernels::call_binary(
                    device.device(),
                    &command_buffer,
                    device.kernels(),
                    hodu_metal_kernels::kernels::$op::I16,
                    &output_shape,
                    lhs,
                    lhs_strides,
                    lhs_offset,
                    rhs,
                    rhs_strides,
                    rhs_offset,
                    &output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?,
                DType::I32 => hodu_metal_kernels::kernels::call_binary(
                    device.device(),
                    &command_buffer,
                    device.kernels(),
                    hodu_metal_kernels::kernels::$op::I32,
                    &output_shape,
                    lhs,
                    lhs_strides,
                    lhs_offset,
                    rhs,
                    rhs_strides,
                    rhs_offset,
                    &output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?,
                DType::I64 => hodu_metal_kernels::kernels::call_binary(
                    device.device(),
                    &command_buffer,
                    device.kernels(),
                    hodu_metal_kernels::kernels::$op::I64,
                    &output_shape,
                    lhs,
                    lhs_strides,
                    lhs_offset,
                    rhs,
                    rhs_strides,
                    rhs_offset,
                    &output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?,
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("comparison '{}'", kernel_name),
                    })
                },
            }
        };
    }

    match kernel_name {
        "eq" => dispatch_cmp!(eq),
        "ne" => dispatch_cmp!(ne),
        "lt" => dispatch_cmp!(lt),
        "le" => dispatch_cmp!(le),
        "gt" => dispatch_cmp!(gt),
        "ge" => dispatch_cmp!(ge),
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: format!("unknown comparison '{}'", kernel_name),
            })
        },
    }

    Ok(MetalStorage::new(output, device.clone(), output_size, output_dtype))
}

pub fn cmp_scalar_map(
    storage: &MetalStorage,
    layout: &Layout,
    scalar: Scalar,
    kernel_name: &str,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::utils::BufferOffset;

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();

    let num_els: usize = shape.iter().product();
    // Output is always BOOL for comparison operations
    let output_dtype = DType::BOOL;
    let output = device.new_buffer(num_els, output_dtype, kernel_name)?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    macro_rules! dispatch_cmp_scalar {
        ($op:ident,$dtype:ident,$ty:ty, $conv:expr) => {{
            let val: $ty = $conv;
            hodu_metal_kernels::kernels::call_unary_scalar(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::$op::$dtype,
                shape,
                input,
                strides,
                offset,
                val,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?
        }};
    }

    match kernel_name {
        "eq_scalar" => match dtype {
            DType::BOOL => dispatch_cmp_scalar!(eq_scalar, BOOL, bool, scalar.to_bool()),
            DType::BF16 => dispatch_cmp_scalar!(eq_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_cmp_scalar!(eq_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_cmp_scalar!(eq_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_cmp_scalar!(eq_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_cmp_scalar!(eq_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_cmp_scalar!(eq_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_cmp_scalar!(eq_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_cmp_scalar!(eq_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_cmp_scalar!(eq_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_cmp_scalar!(eq_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_cmp_scalar!(eq_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "eq_scalar".to_string(),
                })
            },
        },
        "ne_scalar" => match dtype {
            DType::BOOL => dispatch_cmp_scalar!(ne_scalar, BOOL, bool, scalar.to_bool()),
            DType::BF16 => dispatch_cmp_scalar!(ne_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_cmp_scalar!(ne_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_cmp_scalar!(ne_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_cmp_scalar!(ne_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_cmp_scalar!(ne_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_cmp_scalar!(ne_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_cmp_scalar!(ne_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_cmp_scalar!(ne_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_cmp_scalar!(ne_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_cmp_scalar!(ne_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_cmp_scalar!(ne_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "ne_scalar".to_string(),
                })
            },
        },
        "lt_scalar" => match dtype {
            DType::BOOL => dispatch_cmp_scalar!(lt_scalar, BOOL, bool, scalar.to_bool()),
            DType::BF16 => dispatch_cmp_scalar!(lt_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_cmp_scalar!(lt_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_cmp_scalar!(lt_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_cmp_scalar!(lt_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_cmp_scalar!(lt_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_cmp_scalar!(lt_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_cmp_scalar!(lt_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_cmp_scalar!(lt_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_cmp_scalar!(lt_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_cmp_scalar!(lt_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_cmp_scalar!(lt_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "lt_scalar".to_string(),
                })
            },
        },
        "le_scalar" => match dtype {
            DType::BOOL => dispatch_cmp_scalar!(le_scalar, BOOL, bool, scalar.to_bool()),
            DType::BF16 => dispatch_cmp_scalar!(le_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_cmp_scalar!(le_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_cmp_scalar!(le_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_cmp_scalar!(le_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_cmp_scalar!(le_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_cmp_scalar!(le_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_cmp_scalar!(le_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_cmp_scalar!(le_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_cmp_scalar!(le_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_cmp_scalar!(le_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_cmp_scalar!(le_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "le_scalar".to_string(),
                })
            },
        },
        "gt_scalar" => match dtype {
            DType::BOOL => dispatch_cmp_scalar!(gt_scalar, BOOL, bool, scalar.to_bool()),
            DType::BF16 => dispatch_cmp_scalar!(gt_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_cmp_scalar!(gt_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_cmp_scalar!(gt_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_cmp_scalar!(gt_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_cmp_scalar!(gt_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_cmp_scalar!(gt_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_cmp_scalar!(gt_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_cmp_scalar!(gt_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_cmp_scalar!(gt_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_cmp_scalar!(gt_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_cmp_scalar!(gt_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "gt_scalar".to_string(),
                })
            },
        },
        "ge_scalar" => match dtype {
            DType::BOOL => dispatch_cmp_scalar!(ge_scalar, BOOL, bool, scalar.to_bool()),
            DType::BF16 => dispatch_cmp_scalar!(ge_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_cmp_scalar!(ge_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_cmp_scalar!(ge_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_cmp_scalar!(ge_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_cmp_scalar!(ge_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_cmp_scalar!(ge_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_cmp_scalar!(ge_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_cmp_scalar!(ge_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_cmp_scalar!(ge_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_cmp_scalar!(ge_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_cmp_scalar!(ge_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "ge_scalar".to_string(),
                })
            },
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: format!("unknown cmp_scalar operation '{}'", kernel_name),
            })
        },
    }

    Ok(MetalStorage::new(output, device.clone(), num_els, output_dtype))
}

pub fn unary_map(storage: &MetalStorage, layout: &Layout, kernel_name: &str) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::utils::BufferOffset;

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();

    let num_els: usize = shape.iter().product();
    let output = device.new_buffer(num_els, dtype, kernel_name)?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    macro_rules! dispatch_dtype {
        ($op:ident) => {
            match dtype {
                DType::BF16 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    hodu_metal_kernels::kernels::call_unary(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$op::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("unary operation '{}'", kernel_name),
                    });
                },
            }
        };
    }

    match kernel_name {
        "neg" => dispatch_dtype!(neg),
        "abs" => dispatch_dtype!(abs),
        "sign" => dispatch_dtype!(sign),
        "square" => dispatch_dtype!(square),
        "sqrt" => dispatch_dtype!(sqrt),
        "recip" => dispatch_dtype!(recip),
        "relu" => dispatch_dtype!(relu),
        "sigmoid" => dispatch_dtype!(sigmoid),
        "tanh" => dispatch_dtype!(tanh),
        "gelu" => dispatch_dtype!(gelu),
        "softplus" => dispatch_dtype!(softplus),
        "sin" => dispatch_dtype!(sin),
        "cos" => dispatch_dtype!(cos),
        "tan" => dispatch_dtype!(tan),
        "exp" => dispatch_dtype!(exp),
        "exp2" => dispatch_dtype!(exp2),
        "exp10" => dispatch_dtype!(exp10),
        "ln" => dispatch_dtype!(ln),
        "log2" => dispatch_dtype!(log2),
        "log10" => dispatch_dtype!(log10),
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: format!("unknown unary operation '{}'", kernel_name),
            });
        },
    }

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn unary_logical_map(storage: &MetalStorage, layout: &Layout, kernel_name: &str) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::utils::BufferOffset;

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();

    let num_els: usize = shape.iter().product();
    // Output is always BOOL for logical operations
    let output_dtype = DType::BOOL;
    let output = device.new_buffer(num_els, output_dtype, kernel_name)?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    // Only logical_not is supported
    match kernel_name {
        "logical_not" => match dtype {
            DType::BF16 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::BF16,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::F16 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::F16,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::F32 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::F32,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::U8 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::U8,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::U16 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::U16,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::U32 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::U32,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::U64 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::U64,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::I8 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::I8,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::I16 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::I16,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::I32 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::I32,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            DType::I64 => hodu_metal_kernels::kernels::call_unary(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::logical_not::I64,
                shape,
                input,
                strides,
                offset,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?,
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: format!("logical_not"),
                })
            },
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: format!("unknown unary logical operation '{}'", kernel_name),
            })
        },
    }

    Ok(MetalStorage::new(output, device.clone(), num_els, output_dtype))
}

pub fn unary_scalar_map(
    storage: &MetalStorage,
    layout: &Layout,
    scalar: Scalar,
    kernel_name: &str,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::utils::BufferOffset;

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();

    let num_els: usize = shape.iter().product();
    let output = device.new_buffer(num_els, dtype, kernel_name)?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    macro_rules! dispatch_unary_scalar {
        ($op:ident,$dtype:ident,$ty:ty, $conv:expr) => {{
            let val: $ty = $conv;
            hodu_metal_kernels::kernels::call_unary_scalar(
                device.device(),
                &command_buffer,
                device.kernels(),
                hodu_metal_kernels::kernels::$op::$dtype,
                shape,
                input,
                strides,
                offset,
                val,
                &output,
            )
            .map_err(|e| HoduError::Metal(e.into()))?
        }};
    }

    match kernel_name {
        "add_scalar" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(add_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(add_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(add_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_unary_scalar!(add_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_unary_scalar!(add_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_unary_scalar!(add_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_unary_scalar!(add_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_unary_scalar!(add_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_unary_scalar!(add_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_unary_scalar!(add_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_unary_scalar!(add_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "add_scalar".to_string(),
                })
            },
        },
        "sub_scalar" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(sub_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(sub_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(sub_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_unary_scalar!(sub_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_unary_scalar!(sub_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_unary_scalar!(sub_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_unary_scalar!(sub_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_unary_scalar!(sub_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_unary_scalar!(sub_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_unary_scalar!(sub_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_unary_scalar!(sub_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "sub_scalar".to_string(),
                })
            },
        },
        "mul_scalar" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(mul_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(mul_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(mul_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_unary_scalar!(mul_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_unary_scalar!(mul_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_unary_scalar!(mul_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_unary_scalar!(mul_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_unary_scalar!(mul_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_unary_scalar!(mul_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_unary_scalar!(mul_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_unary_scalar!(mul_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "mul_scalar".to_string(),
                })
            },
        },
        "div_scalar" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(div_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(div_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(div_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_unary_scalar!(div_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_unary_scalar!(div_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_unary_scalar!(div_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_unary_scalar!(div_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_unary_scalar!(div_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_unary_scalar!(div_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_unary_scalar!(div_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_unary_scalar!(div_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "div_scalar".to_string(),
                })
            },
        },
        "pow_scalar" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(pow_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(pow_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(pow_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_unary_scalar!(pow_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_unary_scalar!(pow_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_unary_scalar!(pow_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_unary_scalar!(pow_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_unary_scalar!(pow_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_unary_scalar!(pow_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_unary_scalar!(pow_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_unary_scalar!(pow_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "pow_scalar".to_string(),
                })
            },
        },
        "maximum_scalar" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(maximum_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(maximum_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(maximum_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_unary_scalar!(maximum_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_unary_scalar!(maximum_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_unary_scalar!(maximum_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_unary_scalar!(maximum_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_unary_scalar!(maximum_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_unary_scalar!(maximum_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_unary_scalar!(maximum_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_unary_scalar!(maximum_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "maximum_scalar".to_string(),
                })
            },
        },
        "minimum_scalar" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(minimum_scalar, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(minimum_scalar, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(minimum_scalar, F32, f32, scalar.to_f32()),
            DType::U8 => dispatch_unary_scalar!(minimum_scalar, U8, u8, scalar.to_u8()),
            DType::U16 => dispatch_unary_scalar!(minimum_scalar, U16, u16, scalar.to_u16()),
            DType::U32 => dispatch_unary_scalar!(minimum_scalar, U32, u32, scalar.to_u32()),
            DType::U64 => dispatch_unary_scalar!(minimum_scalar, U64, u64, scalar.to_u64()),
            DType::I8 => dispatch_unary_scalar!(minimum_scalar, I8, i8, scalar.to_i8()),
            DType::I16 => dispatch_unary_scalar!(minimum_scalar, I16, i16, scalar.to_i16()),
            DType::I32 => dispatch_unary_scalar!(minimum_scalar, I32, i32, scalar.to_i32()),
            DType::I64 => dispatch_unary_scalar!(minimum_scalar, I64, i64, scalar.to_i64()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "minimum_scalar".to_string(),
                })
            },
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: format!("unknown unary_scalar operation '{}'", kernel_name),
            })
        },
    }

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn matmul_map(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{
        kernels::{call_matmul, matmul},
        utils::BufferOffset,
    };

    let dtype = lhs_storage.get_dtype();
    let device = lhs_storage.get_hodu_device();
    let lhs_shape = lhs_layout.get_shape();
    let lhs_strides = lhs_layout.get_strides();
    let lhs_offset = lhs_layout.get_offset();
    let rhs_shape = rhs_layout.get_shape();
    let rhs_strides = rhs_layout.get_strides();
    let rhs_offset = rhs_layout.get_offset();
    let lhs_ndim = lhs_shape.len();
    let rhs_ndim = rhs_shape.len();

    if lhs_ndim < 2 || rhs_ndim < 2 {
        return Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
            op: "matmul - both tensors must be at least 2D".to_string(),
        });
    }

    let m = lhs_shape[lhs_ndim - 2];
    let k = lhs_shape[lhs_ndim - 1];
    let rhs_k = rhs_shape[rhs_ndim - 2];
    let n = rhs_shape[rhs_ndim - 1];

    if k != rhs_k {
        return Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
            op: "matmul - inner dimensions must match".to_string(),
        });
    }

    let lhs_batch_ndim = lhs_ndim - 2;
    let rhs_batch_ndim = rhs_ndim - 2;
    let batch_ndim = lhs_batch_ndim.max(rhs_batch_ndim);

    let mut batch_shape = vec![1; batch_ndim];
    for i in 0..batch_ndim {
        let lhs_dim = if i < lhs_batch_ndim {
            lhs_shape[lhs_batch_ndim - 1 - i]
        } else {
            1
        };
        let rhs_dim = if i < rhs_batch_ndim {
            rhs_shape[rhs_batch_ndim - 1 - i]
        } else {
            1
        };

        if lhs_dim != 1 && rhs_dim != 1 && lhs_dim != rhs_dim {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
                op: "matmul - incompatible batch dimensions".to_string(),
            });
        }
        batch_shape[batch_ndim - 1 - i] = lhs_dim.max(rhs_dim);
    }

    let total_batches: usize = batch_shape.iter().product();
    let num_els = total_batches * m * n;

    let output = device.new_buffer(num_els, dtype, "matmul")?;
    let command_buffer = device.command_buffer()?;

    let lhs = BufferOffset {
        buffer: lhs_storage.buffer(),
        offset_in_bytes: lhs_offset * dtype.get_size_in_bytes(),
    };
    let rhs = BufferOffset {
        buffer: rhs_storage.buffer(),
        offset_in_bytes: rhs_offset * dtype.get_size_in_bytes(),
    };

    // Prepare metadata for Metal kernel
    // call_matmul expects: metadata[0] = num_els, metadata[1..] = actual kernel metadata
    // Kernel expects: lhs_ndim, rhs_ndim, batch_ndim, lhs_shape, rhs_shape, batch_shape, lhs_strides, rhs_strides, lhs_offset, rhs_offset, M, K, N
    let mut metadata = Vec::new();
    metadata.push(num_els); // For call_matmul
    metadata.push(lhs_ndim);
    metadata.push(rhs_ndim);
    metadata.push(batch_ndim);
    metadata.extend_from_slice(lhs_shape);
    metadata.extend_from_slice(rhs_shape);
    metadata.extend_from_slice(&batch_shape);
    metadata.extend_from_slice(lhs_strides);
    metadata.extend_from_slice(rhs_strides);
    metadata.push(lhs_offset);
    metadata.push(rhs_offset);
    metadata.push(m);
    metadata.push(k);
    metadata.push(n);

    match dtype {
        DType::BF16 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::BF16,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::F16 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::F16,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::F32 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::F32,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::U8 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::U8,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::U16 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::U16,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::U32 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::U32,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::U64 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::U64,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::I8 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::I8,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::I16 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::I16,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::I32 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::I32,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::I64 => {
            call_matmul(
                device.device(),
                &command_buffer,
                device.kernels(),
                matmul::I64,
                lhs,
                rhs,
                &output,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: "matmul".to_string(),
            })
        },
    }

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn dot_map(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{
        kernels::{call_dot, dot},
        utils::BufferOffset,
    };

    let dtype = lhs_storage.get_dtype();
    let device = lhs_storage.get_hodu_device();
    let lhs_shape = lhs_layout.get_shape();
    let lhs_strides = lhs_layout.get_strides();
    let lhs_offset = lhs_layout.get_offset();
    let rhs_shape = rhs_layout.get_shape();
    let rhs_strides = rhs_layout.get_strides();
    let rhs_offset = rhs_layout.get_offset();

    if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
        return Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
            op: "dot - only 2D tensors supported".to_string(),
        });
    }

    let (m, k1) = (lhs_shape[0], lhs_shape[1]);
    let (k2, n) = (rhs_shape[0], rhs_shape[1]);

    if k1 != k2 {
        return Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
            op: "dot - inner dimensions must match".to_string(),
        });
    }

    let num_els = m * n;

    let output = device.new_buffer(num_els, dtype, "dot")?;
    let command_buffer = device.command_buffer()?;

    let lhs = BufferOffset {
        buffer: lhs_storage.buffer(),
        offset_in_bytes: lhs_offset * dtype.get_size_in_bytes(),
    };
    let rhs = BufferOffset {
        buffer: rhs_storage.buffer(),
        offset_in_bytes: rhs_offset * dtype.get_size_in_bytes(),
    };

    // Prepare metadata for Metal kernel
    // Layout: [M, K, unused, N, lhs_stride_m, lhs_stride_k, rhs_stride_k, rhs_stride_n, lhs_offset, rhs_offset]
    // Note: metadata[2] is unused by the kernel, but metadata[3] is N
    let metadata = vec![
        m,
        k1,
        0, // unused (metadata[2])
        n, // metadata[3]
        lhs_strides[0],
        lhs_strides[1],
        rhs_strides[0],
        rhs_strides[1],
        lhs_offset,
        rhs_offset,
    ];

    match dtype {
        DType::BF16 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::BF16,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::F16 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::F16,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::F32 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::F32,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::U8 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::U8,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::U16 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::U16,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::U32 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::U32,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::U64 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::U64,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::I8 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::I8,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::I16 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::I16,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::I32 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::I32,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        DType::I64 => {
            call_dot(
                device.device(),
                &command_buffer,
                device.kernels(),
                dot::I64,
                lhs,
                rhs,
                &output,
                m,
                n,
                &metadata,
            )
            .map_err(|e| HoduError::Metal(e.into()))?;
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: "dot".to_string(),
            })
        },
    }

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}
