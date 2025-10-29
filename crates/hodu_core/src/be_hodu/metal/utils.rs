#![allow(clippy::vec_init_then_push)]

use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    op::{
        conv::{
            ParamsConv1D, ParamsConv2D, ParamsConv3D, ParamsConvTranspose1D, ParamsConvTranspose2D,
            ParamsConvTranspose3D,
        },
        window_reduction::WindowReduction,
        ReduceOp,
    },
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
        "silu" => dispatch_dtype!(silu),
        "mish" => dispatch_dtype!(mish),
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
                    op: "logical_not".to_string(),
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
        ($op:ident, $dtype:ident, $ty:ty, $conv:expr) => {{
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
        "leaky_relu" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(leaky_relu, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(leaky_relu, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(leaky_relu, F32, f32, scalar.to_f32()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "leaky_relu".to_string(),
                })
            },
        },
        "elu" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(elu, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(elu, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(elu, F32, f32, scalar.to_f32()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "elu".to_string(),
                })
            },
        },
        "prelu" => match dtype {
            DType::BF16 => dispatch_unary_scalar!(prelu, BF16, half::bf16, scalar.to_bf16()),
            DType::F16 => dispatch_unary_scalar!(prelu, F16, half::f16, scalar.to_f16()),
            DType::F32 => dispatch_unary_scalar!(prelu, F32, f32, scalar.to_f32()),
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype,
                    op: "prelu".to_string(),
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

pub fn concat_map(
    first: &MetalStorage,
    others: &[&MetalStorage],
    layouts: &[&Layout],
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_concat, utils::BufferOffset};

    let dtype = first.get_dtype();
    let device = first.get_hodu_device();

    // Validate all storages have same dtype
    for other in others {
        if other.get_dtype() != dtype {
            return Err(HoduError::DTypeConflictInOp {
                left: dtype,
                right: other.get_dtype(),
                op: "concat".to_string(),
            });
        }
    }

    let first_shape = layouts[0].get_shape();
    let ndim = first_shape.len();

    if dim >= ndim {
        return Err(HoduError::IncompatibleShapes {
            lhs: first_shape.to_vec(),
            rhs: vec![],
            op: format!(
                "concat - dimension {} out of range for {}-dimensional tensor",
                dim, ndim
            ),
        });
    }

    // Verify all tensors have same shape except at concat dimension
    for layout in layouts.iter().skip(1) {
        let shape = layout.get_shape();
        if shape.len() != ndim {
            return Err(HoduError::IncompatibleShapes {
                lhs: first_shape.to_vec(),
                rhs: shape.to_vec(),
                op: "concat - all tensors must have the same number of dimensions".to_string(),
            });
        }
        for (j, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
            if j != dim && s1 != s2 {
                return Err(HoduError::IncompatibleShapes {
                    lhs: first_shape.to_vec(),
                    rhs: shape.to_vec(),
                    op: format!("concat - dimension {} must match (got {} vs {})", j, s1, s2),
                });
            }
        }
    }

    // Calculate output shape
    let mut output_shape = first_shape.to_vec();
    output_shape[dim] = layouts.iter().map(|l| l.get_shape()[dim]).sum();

    let num_els: usize = output_shape.iter().product();
    let output = device.new_buffer(num_els, dtype, "concat")?;
    let command_buffer = device.command_buffer()?;

    // Prepare input shapes, strides, offsets
    let mut input_shapes = Vec::with_capacity(layouts.len() * ndim);
    let mut input_strides = Vec::with_capacity(layouts.len() * ndim);
    let mut input_offsets = Vec::with_capacity(layouts.len());
    let mut input_buffer_offsets = Vec::with_capacity(layouts.len());

    // Collect all inputs (first + others)
    let all_storages: Vec<&MetalStorage> = std::iter::once(first).chain(others.iter().copied()).collect();

    // Calculate total size needed for combined buffer
    let mut total_elements = 0;
    for (_storage, layout) in all_storages.iter().zip(layouts.iter()) {
        let shape = layout.get_shape();
        let num_els_in_storage: usize = shape.iter().product();
        total_elements += num_els_in_storage;
    }

    // Create a combined buffer that holds all input tensors
    let combined_buffer = device.new_buffer(total_elements, dtype, "concat_combined_input")?;

    // Copy all input tensors into the combined buffer and track offsets
    let mut cumulative_offset = 0;
    let encoder = command_buffer.blit_command_encoder();

    for (storage, layout) in all_storages.iter().zip(layouts.iter()) {
        let shape = layout.get_shape();
        let strides = layout.get_strides();
        let offset = layout.get_offset();
        let num_els_in_storage: usize = shape.iter().product();

        input_shapes.extend_from_slice(shape);
        input_strides.extend_from_slice(strides);
        input_offsets.push(offset);
        input_buffer_offsets.push(cumulative_offset);

        // Copy this storage's buffer to the combined buffer
        let source_offset = offset * dtype.get_size_in_bytes();
        let dest_offset = cumulative_offset * dtype.get_size_in_bytes();
        let size = num_els_in_storage * dtype.get_size_in_bytes();

        encoder.copy_from_buffer(storage.buffer(), source_offset, &combined_buffer, dest_offset, size);

        cumulative_offset += num_els_in_storage;
    }
    encoder.end_encoding();

    let input = BufferOffset {
        buffer: &combined_buffer,
        offset_in_bytes: 0,
    };

    macro_rules! dispatch_concat {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BOOL => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BOOL,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::BF16 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_concat(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        &output_shape,
                        dim,
                        &input_shapes,
                        &input_strides,
                        &input_offsets,
                        &input_buffer_offsets,
                        input,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "concat".to_string(),
                    })
                },
            }
        };
    }

    dispatch_concat!(concat);

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn split_map(
    storage: &MetalStorage,
    layout: &Layout,
    dim: usize,
    sizes: &[usize],
) -> HoduResult<Vec<MetalStorage>> {
    use hodu_metal_kernels::{kernels::call_split, utils::BufferOffset};

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::IncompatibleShapes {
            lhs: shape.to_vec(),
            rhs: vec![],
            op: format!("split - dimension {} out of range for {}-dimensional tensor", dim, ndim),
        });
    }

    // Verify sizes sum to dimension size
    let total_size: usize = sizes.iter().sum();
    if total_size != shape[dim] {
        return Err(HoduError::IncompatibleShapes {
            lhs: vec![shape[dim]],
            rhs: vec![total_size],
            op: format!(
                "split - sizes must sum to dimension size (got {} vs {})",
                total_size, shape[dim]
            ),
        });
    }

    let command_buffer = device.command_buffer()?;
    let input_buffer = storage.buffer();
    let input_offset_bytes = offset * dtype.get_size_in_bytes();

    let mut results = Vec::with_capacity(sizes.len());
    let mut split_offset = 0;

    for &size in sizes {
        let mut output_shape = shape.to_vec();
        output_shape[dim] = size;
        let num_els: usize = output_shape.iter().product();

        let output = device.new_buffer(num_els, dtype, "split")?;

        let input = BufferOffset {
            buffer: input_buffer,
            offset_in_bytes: input_offset_bytes,
        };

        macro_rules! dispatch_split {
            ($kernel_mod:ident) => {
                match dtype {
                    DType::BOOL => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::BOOL,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::BF16 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::BF16,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::F16 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::F16,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::F32 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::F32,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::U8 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::U8,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::U16 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::U16,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::U32 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::U32,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::U64 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::U64,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::I8 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::I8,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::I16 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::I16,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::I32 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::I32,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    DType::I64 => {
                        call_split(
                            device.device(),
                            &command_buffer,
                            device.kernels(),
                            hodu_metal_kernels::kernels::$kernel_mod::I64,
                            shape,
                            input,
                            strides,
                            offset,
                            dim,
                            size,
                            split_offset,
                            &output,
                        )
                        .map_err(|e| HoduError::Metal(e.into()))?;
                    },
                    _ => {
                        return Err(HoduError::UnsupportedDType {
                            dtype,
                            op: "split".to_string(),
                        })
                    },
                }
            };
        }

        dispatch_split!(split);

        results.push(MetalStorage::new(output, device.clone(), num_els, dtype));
        split_offset += size;
    }

    Ok(results)
}

pub fn index_select_map(
    storage: &MetalStorage,
    layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_index_select, utils::BufferOffset};

    // Convert indices to I32 if needed
    let indices_i32 = match indices_storage.get_dtype() {
        DType::I32 => indices_storage.clone(),
        DType::I64 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::U8 | DType::U16 => {
            // Need to convert to I32
            let indices_cpu = indices_storage.to_cpu_storage()?;
            let converted_cpu = match indices_cpu {
                crate::be_hodu::cpu::storage::CpuStorage::I32(_) => indices_cpu,
                crate::be_hodu::cpu::storage::CpuStorage::I64(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U32(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U64(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::I8(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::I16(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U8(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U16(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype: indices_storage.get_dtype(),
                        op: "index_select - indices must be integer type".to_string(),
                    })
                },
            };
            MetalStorage::from_cpu_storage(&converted_cpu)?
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype: indices_storage.get_dtype(),
                op: "index_select - indices must be integer type".to_string(),
            })
        },
    };

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::IncompatibleShapes {
            lhs: shape.to_vec(),
            rhs: vec![],
            op: format!(
                "index_select - dimension {} out of range for {}-dimensional tensor",
                dim, ndim
            ),
        });
    }

    let num_indices: usize = indices_layout.get_shape().iter().product();

    // Calculate output shape
    let mut output_shape = shape.to_vec();
    output_shape[dim] = num_indices;
    let num_els: usize = output_shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "index_select")?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    let indices = BufferOffset {
        buffer: indices_i32.buffer(),
        offset_in_bytes: indices_layout.get_offset() * DType::I32.get_size_in_bytes(),
    };

    macro_rules! dispatch_index_select {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BOOL => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BOOL,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::BF16 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_index_select(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "index_select".to_string(),
                    })
                },
            }
        };
    }

    dispatch_index_select!(index_select);

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn index_put_map(
    storage: &MetalStorage,
    layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    values_storage: &MetalStorage,
    values_layout: &Layout,
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_index_put, utils::BufferOffset};

    // Convert indices to I32 if needed
    let indices_i32 = match indices_storage.get_dtype() {
        DType::I32 => indices_storage.clone(),
        DType::I64 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::U8 | DType::U16 => {
            let indices_cpu = indices_storage.to_cpu_storage()?;
            let converted_cpu = match indices_cpu {
                crate::be_hodu::cpu::storage::CpuStorage::I32(_) => indices_cpu,
                crate::be_hodu::cpu::storage::CpuStorage::I64(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U32(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U64(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::I8(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::I16(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U8(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U16(data) => {
                    let converted: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I32(converted)
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype: indices_storage.get_dtype(),
                        op: "index_put - indices must be integer type".to_string(),
                    })
                },
            };
            MetalStorage::from_cpu_storage(&converted_cpu)?
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype: indices_storage.get_dtype(),
                op: "index_put - indices must be integer type".to_string(),
            })
        },
    };

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dtype != values_storage.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: values_storage.get_dtype(),
            op: "index_put".to_string(),
        });
    }

    if dim >= ndim {
        return Err(HoduError::IncompatibleShapes {
            lhs: shape.to_vec(),
            rhs: vec![],
            op: format!(
                "index_put - dimension {} out of range for {}-dimensional tensor",
                dim, ndim
            ),
        });
    }

    let num_indices: usize = indices_layout.get_shape().iter().product();
    let num_els: usize = shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "index_put")?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    let indices = BufferOffset {
        buffer: indices_i32.buffer(),
        offset_in_bytes: indices_layout.get_offset() * DType::I32.get_size_in_bytes(),
    };

    let values = BufferOffset {
        buffer: values_storage.buffer(),
        offset_in_bytes: values_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let values_strides = values_layout.get_strides();
    let values_offset = values_layout.get_offset();

    macro_rules! dispatch_index_put {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BOOL => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BOOL,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::BF16 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_index_put(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        values,
                        values_strides,
                        values_offset,
                        dim,
                        num_indices,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "index_put".to_string(),
                    })
                },
            }
        };
    }

    dispatch_index_put!(index_put);

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn gather_map(
    storage: &MetalStorage,
    layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_gather, utils::BufferOffset};

    // Convert indices to I64 if needed
    let indices_i64 = match indices_storage.get_dtype() {
        DType::I64 => indices_storage.clone(),
        DType::I32 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::U8 | DType::U16 => {
            let indices_cpu = indices_storage.to_cpu_storage()?;
            let converted_cpu = match indices_cpu {
                crate::be_hodu::cpu::storage::CpuStorage::I64(_) => indices_cpu,
                crate::be_hodu::cpu::storage::CpuStorage::I32(data) => {
                    let converted: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I64(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U32(data) => {
                    let converted: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I64(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U64(data) => {
                    let converted: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I64(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::I8(data) => {
                    let converted: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I64(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::I16(data) => {
                    let converted: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I64(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U8(data) => {
                    let converted: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I64(converted)
                },
                crate::be_hodu::cpu::storage::CpuStorage::U16(data) => {
                    let converted: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                    crate::be_hodu::cpu::storage::CpuStorage::I64(converted)
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype: indices_storage.get_dtype(),
                        op: "gather - indices must be integer type".to_string(),
                    })
                },
            };
            MetalStorage::from_cpu_storage(&converted_cpu)?
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype: indices_storage.get_dtype(),
                op: "gather - indices must be integer type".to_string(),
            })
        },
    };

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(HoduError::IncompatibleShapes {
            lhs: shape.to_vec(),
            rhs: vec![],
            op: format!(
                "gather - dimension {} out of range for {}-dimensional tensor",
                dim, ndim
            ),
        });
    }

    let indices_shape = indices_layout.get_shape();
    let num_els: usize = indices_shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "gather")?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    let indices = BufferOffset {
        buffer: indices_i64.buffer(),
        offset_in_bytes: indices_layout.get_offset() * DType::I64.get_size_in_bytes(),
    };

    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    macro_rules! dispatch_gather {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BOOL => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BOOL,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::BF16 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_gather(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "gather".to_string(),
                    })
                },
            }
        };
    }

    dispatch_gather!(gather);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn scatter_map(
    storage: &MetalStorage,
    layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    src_storage: &MetalStorage,
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_scatter, utils::BufferOffset};

    let indices_i64 = match indices_storage.get_dtype() {
        DType::I64 => indices_storage.clone(),
        DType::I32 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::U8 | DType::U16 => {
            let indices_cpu = indices_storage.to_cpu_storage()?;
            let converted_cpu = match indices_cpu {
                crate::be_hodu::cpu::storage::CpuStorage::I64(_) => indices_cpu,
                crate::be_hodu::cpu::storage::CpuStorage::I32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U64(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype: indices_storage.get_dtype(),
                        op: "scatter - indices must be integer type".to_string(),
                    })
                },
            };
            MetalStorage::from_cpu_storage(&converted_cpu)?
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype: indices_storage.get_dtype(),
                op: "scatter - indices must be integer type".to_string(),
            })
        },
    };

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();

    if dtype != src_storage.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: src_storage.get_dtype(),
            op: "scatter".to_string(),
        });
    }

    let num_els: usize = shape.iter().product();
    let output = device.new_buffer(num_els, dtype, "scatter")?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    let indices = BufferOffset {
        buffer: indices_i64.buffer(),
        offset_in_bytes: indices_layout.get_offset() * DType::I64.get_size_in_bytes(),
    };

    let src = BufferOffset {
        buffer: src_storage.buffer(),
        offset_in_bytes: src_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    macro_rules! dispatch_scatter {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BOOL => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BOOL,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::BF16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "scatter".to_string(),
                    })
                },
            }
        };
    }

    dispatch_scatter!(scatter);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn scatter_add_map(
    storage: &MetalStorage,
    layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    src_storage: &MetalStorage,
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_scatter, utils::BufferOffset};

    let indices_i64 = match indices_storage.get_dtype() {
        DType::I64 => indices_storage.clone(),
        DType::I32 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::U8 | DType::U16 => {
            let indices_cpu = indices_storage.to_cpu_storage()?;
            let converted_cpu = match indices_cpu {
                crate::be_hodu::cpu::storage::CpuStorage::I64(_) => indices_cpu,
                crate::be_hodu::cpu::storage::CpuStorage::I32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U64(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype: indices_storage.get_dtype(),
                        op: "scatter_add - indices must be integer type".to_string(),
                    })
                },
            };
            MetalStorage::from_cpu_storage(&converted_cpu)?
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype: indices_storage.get_dtype(),
                op: "scatter_add - indices must be integer type".to_string(),
            })
        },
    };

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();

    if dtype != src_storage.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: src_storage.get_dtype(),
            op: "scatter_add".to_string(),
        });
    }

    let num_els: usize = shape.iter().product();
    let output = device.new_buffer(num_els, dtype, "scatter_add")?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    let indices = BufferOffset {
        buffer: indices_i64.buffer(),
        offset_in_bytes: indices_layout.get_offset() * DType::I64.get_size_in_bytes(),
    };

    let src = BufferOffset {
        buffer: src_storage.buffer(),
        offset_in_bytes: src_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    macro_rules! dispatch_scatter_add {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "scatter_add".to_string(),
                    })
                },
            }
        };
    }

    dispatch_scatter_add!(scatter_add);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn scatter_max_map(
    storage: &MetalStorage,
    layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    src_storage: &MetalStorage,
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_scatter, utils::BufferOffset};

    let indices_i64 = match indices_storage.get_dtype() {
        DType::I64 => indices_storage.clone(),
        DType::I32 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::U8 | DType::U16 => {
            let indices_cpu = indices_storage.to_cpu_storage()?;
            let converted_cpu = match indices_cpu {
                crate::be_hodu::cpu::storage::CpuStorage::I64(_) => indices_cpu,
                crate::be_hodu::cpu::storage::CpuStorage::I32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U64(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype: indices_storage.get_dtype(),
                        op: "scatter_max - indices must be integer type".to_string(),
                    })
                },
            };
            MetalStorage::from_cpu_storage(&converted_cpu)?
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype: indices_storage.get_dtype(),
                op: "scatter_max - indices must be integer type".to_string(),
            })
        },
    };

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();

    if dtype != src_storage.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: src_storage.get_dtype(),
            op: "scatter_max".to_string(),
        });
    }

    let num_els: usize = shape.iter().product();
    let output = device.new_buffer(num_els, dtype, "scatter_max")?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    let indices = BufferOffset {
        buffer: indices_i64.buffer(),
        offset_in_bytes: indices_layout.get_offset() * DType::I64.get_size_in_bytes(),
    };

    let src = BufferOffset {
        buffer: src_storage.buffer(),
        offset_in_bytes: src_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    macro_rules! dispatch_scatter_max {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "scatter_max".to_string(),
                    })
                },
            }
        };
    }

    dispatch_scatter_max!(scatter_max);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn scatter_min_map(
    storage: &MetalStorage,
    layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    src_storage: &MetalStorage,
    src_layout: &Layout,
    dim: usize,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_scatter, utils::BufferOffset};

    let indices_i64 = match indices_storage.get_dtype() {
        DType::I64 => indices_storage.clone(),
        DType::I32 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::U8 | DType::U16 => {
            let indices_cpu = indices_storage.to_cpu_storage()?;
            let converted_cpu = match indices_cpu {
                crate::be_hodu::cpu::storage::CpuStorage::I64(_) => indices_cpu,
                crate::be_hodu::cpu::storage::CpuStorage::I32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U32(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U64(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::I16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U8(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                crate::be_hodu::cpu::storage::CpuStorage::U16(data) => {
                    crate::be_hodu::cpu::storage::CpuStorage::I64(data.iter().map(|&v| v as i64).collect())
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype: indices_storage.get_dtype(),
                        op: "scatter_min - indices must be integer type".to_string(),
                    })
                },
            };
            MetalStorage::from_cpu_storage(&converted_cpu)?
        },
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype: indices_storage.get_dtype(),
                op: "scatter_min - indices must be integer type".to_string(),
            })
        },
    };

    let dtype = storage.get_dtype();
    let device = storage.get_hodu_device();
    let shape = layout.get_shape();
    let strides = layout.get_strides();
    let offset = layout.get_offset();

    if dtype != src_storage.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: src_storage.get_dtype(),
            op: "scatter_min".to_string(),
        });
    }

    let num_els: usize = shape.iter().product();
    let output = device.new_buffer(num_els, dtype, "scatter_min")?;
    let command_buffer = device.command_buffer()?;

    let input = BufferOffset {
        buffer: storage.buffer(),
        offset_in_bytes: offset * dtype.get_size_in_bytes(),
    };

    let indices = BufferOffset {
        buffer: indices_i64.buffer(),
        offset_in_bytes: indices_layout.get_offset() * DType::I64.get_size_in_bytes(),
    };

    let src = BufferOffset {
        buffer: src_storage.buffer(),
        offset_in_bytes: src_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let src_shape = src_layout.get_shape();
    let src_strides = src_layout.get_strides();
    let src_offset = src_layout.get_offset();
    let indices_strides = indices_layout.get_strides();
    let indices_offset = indices_layout.get_offset();

    macro_rules! dispatch_scatter_min {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_scatter(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        shape,
                        input,
                        strides,
                        offset,
                        indices,
                        indices_strides,
                        indices_offset,
                        src,
                        src_shape,
                        src_strides,
                        src_offset,
                        dim,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "scatter_min".to_string(),
                    })
                },
            }
        };
    }

    dispatch_scatter_min!(scatter_min);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn conv1d_map(
    input: &MetalStorage,
    weight: &MetalStorage,
    input_layout: &Layout,
    weight_layout: &Layout,
    params: &ParamsConv1D,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_conv, utils::BufferOffset};

    let dtype = input.get_dtype();
    let device = input.get_hodu_device();

    if dtype != weight.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: weight.get_dtype(),
            op: "conv1d".to_string(),
        });
    }

    let input_shape = input_layout.get_shape();
    let weight_shape = weight_layout.get_shape();

    // Input: [batch, in_channels, length]
    // Weight: [out_channels, in_channels, kernel_size]
    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let input_length = input_shape[2];

    let out_channels = weight_shape[0];
    let kernel_size = weight_shape[2];

    // Calculate output length
    let output_length =
        (input_length + 2 * params.padding - params.dilation * (kernel_size - 1) - 1) / params.stride + 1;

    let output_shape = [batch, out_channels, output_length];
    let num_els: usize = output_shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "conv1d")?;
    let command_buffer = device.command_buffer()?;

    let input_buf = BufferOffset {
        buffer: input.buffer(),
        offset_in_bytes: input_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let weight_buf = BufferOffset {
        buffer: weight.buffer(),
        offset_in_bytes: weight_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    // Metadata: [num_els, batch, in_channels, out_channels, input_length, output_length, kernel_size, stride, padding, dilation]
    let metadata = vec![
        num_els,
        batch,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        params.stride,
        params.padding,
        params.dilation,
    ];

    macro_rules! dispatch_conv1d {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "conv1d".to_string(),
                    })
                },
            }
        };
    }

    dispatch_conv1d!(conv1d);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn conv2d_map(
    input: &MetalStorage,
    weight: &MetalStorage,
    input_layout: &Layout,
    weight_layout: &Layout,
    params: &ParamsConv2D,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_conv, utils::BufferOffset};

    let dtype = input.get_dtype();
    let device = input.get_hodu_device();

    if dtype != weight.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: weight.get_dtype(),
            op: "conv2d".to_string(),
        });
    }

    let input_shape = input_layout.get_shape();
    let weight_shape = weight_layout.get_shape();

    // Input: [batch, in_channels, height, width]
    // Weight: [out_channels, in_channels, kernel_h, kernel_w]
    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let input_h = input_shape[2];
    let input_w = input_shape[3];

    let out_channels = weight_shape[0];
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];

    // Calculate output dimensions
    let output_h = (input_h + 2 * params.padding - params.dilation * (kernel_h - 1) - 1) / params.stride + 1;
    let output_w = (input_w + 2 * params.padding - params.dilation * (kernel_w - 1) - 1) / params.stride + 1;

    let output_shape = [batch, out_channels, output_h, output_w];
    let num_els: usize = output_shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "conv2d")?;
    let command_buffer = device.command_buffer()?;

    let input_buf = BufferOffset {
        buffer: input.buffer(),
        offset_in_bytes: input_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let weight_buf = BufferOffset {
        buffer: weight.buffer(),
        offset_in_bytes: weight_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    // Metadata: [num_els, batch, in_channels, out_channels, input_h, input_w, output_h, output_w,
    //           kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w]
    let metadata = vec![
        num_els,
        batch,
        in_channels,
        out_channels,
        input_h,
        input_w,
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        params.stride,
        params.stride,
        params.padding,
        params.padding,
        params.dilation,
        params.dilation,
    ];

    macro_rules! dispatch_conv2d {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "conv2d".to_string(),
                    })
                },
            }
        };
    }

    dispatch_conv2d!(conv2d);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

pub fn conv3d_map(
    input: &MetalStorage,
    weight: &MetalStorage,
    input_layout: &Layout,
    weight_layout: &Layout,
    params: &ParamsConv3D,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_conv, utils::BufferOffset};

    let dtype = input.get_dtype();
    let device = input.get_hodu_device();

    if dtype != weight.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: weight.get_dtype(),
            op: "conv3d".to_string(),
        });
    }

    let input_shape = input_layout.get_shape();
    let weight_shape = weight_layout.get_shape();

    // Input: [batch, in_channels, depth, height, width]
    // Weight: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let input_d = input_shape[2];
    let input_h = input_shape[3];
    let input_w = input_shape[4];

    let out_channels = weight_shape[0];
    let kernel_d = weight_shape[2];
    let kernel_h = weight_shape[3];
    let kernel_w = weight_shape[4];

    // Calculate output dimensions
    let output_d = (input_d + 2 * params.padding - params.dilation * (kernel_d - 1) - 1) / params.stride + 1;
    let output_h = (input_h + 2 * params.padding - params.dilation * (kernel_h - 1) - 1) / params.stride + 1;
    let output_w = (input_w + 2 * params.padding - params.dilation * (kernel_w - 1) - 1) / params.stride + 1;

    let output_shape = [batch, out_channels, output_d, output_h, output_w];
    let num_els: usize = output_shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "conv3d")?;
    let command_buffer = device.command_buffer()?;

    let input_buf = BufferOffset {
        buffer: input.buffer(),
        offset_in_bytes: input_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let weight_buf = BufferOffset {
        buffer: weight.buffer(),
        offset_in_bytes: weight_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    // Metadata: [num_els, batch, in_channels, out_channels, input_d, input_h, input_w,
    //           output_d, output_h, output_w, kernel_d, kernel_h, kernel_w,
    //           stride_d, stride_h, stride_w, padding_d, padding_h, padding_w,
    //           dilation_d, dilation_h, dilation_w]
    let metadata = vec![
        num_els,
        batch,
        in_channels,
        out_channels,
        input_d,
        input_h,
        input_w,
        output_d,
        output_h,
        output_w,
        kernel_d,
        kernel_h,
        kernel_w,
        params.stride,
        params.stride,
        params.stride,
        params.padding,
        params.padding,
        params.padding,
        params.dilation,
        params.dilation,
        params.dilation,
    ];

    macro_rules! dispatch_conv3d {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "conv3d".to_string(),
                    })
                },
            }
        };
    }

    dispatch_conv3d!(conv3d);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

/// Transpose convolution 1D operation
pub(crate) fn conv_transpose1d_map(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    weight_storage: &MetalStorage,
    weight_layout: &Layout,
    params: &ParamsConvTranspose1D,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_conv, utils::BufferOffset};

    let device = input_storage.get_hodu_device();
    let dtype = input_storage.get_dtype();

    // Calculate output length: (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    let output_length = (params.length_input - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_size - 1)
        + params.output_padding
        + 1;

    let num_els = params.batch_size * params.channels_output * output_length;

    // Metadata: [num_els, batch_size, in_channels, out_channels, input_length, output_length,
    //            kernel_size, padding, output_padding, stride, dilation]
    let metadata = vec![
        num_els,
        params.batch_size,
        params.channels_input,
        params.channels_output,
        params.length_input,
        output_length,
        params.kernel_size,
        params.padding,
        params.output_padding,
        params.stride,
        params.dilation,
    ];

    let output = device.new_buffer(num_els, dtype, "conv_transpose1d_output")?;
    let command_buffer = device.command_buffer()?;

    let input_buf = BufferOffset {
        buffer: input_storage.buffer(),
        offset_in_bytes: input_layout.get_offset() * dtype.get_size_in_bytes(),
    };
    let weight_buf = BufferOffset {
        buffer: weight_storage.buffer(),
        offset_in_bytes: weight_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    macro_rules! dispatch_conv_transpose1d {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "conv_transpose1d".to_string(),
                    })
                },
            }
        };
    }

    dispatch_conv_transpose1d!(conv_transpose1d);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

/// Transpose convolution 2D operation
pub(crate) fn conv_transpose2d_map(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    weight_storage: &MetalStorage,
    weight_layout: &Layout,
    params: &ParamsConvTranspose2D,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_conv, utils::BufferOffset};

    let device = input_storage.get_hodu_device();
    let dtype = input_storage.get_dtype();

    // Calculate output dimensions
    let output_height = (params.input_height - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_height - 1)
        + params.output_padding
        + 1;
    let output_width = (params.input_width - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_width - 1)
        + params.output_padding
        + 1;

    let num_els = params.batch_size * params.channels_output * output_height * output_width;

    // Metadata: [num_els, batch_size, in_channels, out_channels, input_height, input_width,
    //            output_height, output_width, kernel_height, kernel_width, padding, output_padding, stride, dilation]
    let metadata = vec![
        num_els,
        params.batch_size,
        params.channels_input,
        params.channels_output,
        params.input_height,
        params.input_width,
        output_height,
        output_width,
        params.kernel_height,
        params.kernel_width,
        params.padding,
        params.output_padding,
        params.stride,
        params.dilation,
    ];

    let output = device.new_buffer(num_els, dtype, "conv_transpose2d_output")?;
    let command_buffer = device.command_buffer()?;

    let input_buf = BufferOffset {
        buffer: input_storage.buffer(),
        offset_in_bytes: input_layout.get_offset() * dtype.get_size_in_bytes(),
    };
    let weight_buf = BufferOffset {
        buffer: weight_storage.buffer(),
        offset_in_bytes: weight_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    macro_rules! dispatch_conv_transpose2d {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "conv_transpose2d".to_string(),
                    })
                },
            }
        };
    }

    dispatch_conv_transpose2d!(conv_transpose2d);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

/// Transpose convolution 3D operation
pub(crate) fn conv_transpose3d_map(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    weight_storage: &MetalStorage,
    weight_layout: &Layout,
    params: &ParamsConvTranspose3D,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_conv, utils::BufferOffset};

    let device = input_storage.get_hodu_device();
    let dtype = input_storage.get_dtype();

    // Calculate output dimensions
    let output_depth = (params.input_depth - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_depth - 1)
        + params.output_padding
        + 1;
    let output_height = (params.input_height - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_height - 1)
        + params.output_padding
        + 1;
    let output_width = (params.input_width - 1) * params.stride - 2 * params.padding
        + params.dilation * (params.kernel_width - 1)
        + params.output_padding
        + 1;

    let num_els = params.batch_size * params.channels_output * output_depth * output_height * output_width;

    // Metadata: [num_els, batch_size, in_channels, out_channels, input_depth, input_height, input_width,
    //            output_depth, output_height, output_width, kernel_depth, kernel_height, kernel_width,
    //            padding, output_padding, stride, dilation]
    let metadata = vec![
        num_els,
        params.batch_size,
        params.channels_input,
        params.channels_output,
        params.input_depth,
        params.input_height,
        params.input_width,
        output_depth,
        output_height,
        output_width,
        params.kernel_depth,
        params.kernel_height,
        params.kernel_width,
        params.padding,
        params.output_padding,
        params.stride,
        params.dilation,
    ];

    let output = device.new_buffer(num_els, dtype, "conv_transpose3d_output")?;
    let command_buffer = device.command_buffer()?;

    let input_buf = BufferOffset {
        buffer: input_storage.buffer(),
        offset_in_bytes: input_layout.get_offset() * dtype.get_size_in_bytes(),
    };
    let weight_buf = BufferOffset {
        buffer: weight_storage.buffer(),
        offset_in_bytes: weight_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    macro_rules! dispatch_conv_transpose3d {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_conv(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        input_buf,
                        weight_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "conv_transpose3d".to_string(),
                    })
                },
            }
        };
    }

    dispatch_conv_transpose3d!(conv_transpose3d);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

/// Convolution 1D gradient weight operation
pub(crate) fn conv1d_grad_weight_map(
    input: &MetalStorage,
    grad_output: &MetalStorage,
    input_layout: &Layout,
    grad_output_layout: &Layout,
    params: &ParamsConv1D,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_conv_grad_weight, utils::BufferOffset};

    let dtype = input.get_dtype();
    let device = input.get_hodu_device();

    if dtype != grad_output.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: grad_output.get_dtype(),
            op: "conv1d_grad_weight".to_string(),
        });
    }

    let input_shape = input_layout.get_shape();
    let grad_output_shape = grad_output_layout.get_shape();

    // Input: [batch, in_channels, length]
    // Grad Output: [batch, out_channels, output_length]
    // Grad Weight: [out_channels, in_channels, kernel_size]
    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let input_length = input_shape[2];

    let out_channels = grad_output_shape[1];
    let output_length = grad_output_shape[2];

    let kernel_size = params.kernel_size;

    // Output is the grad_weight with shape [out_channels, in_channels, kernel_size]
    let grad_weight_shape = [out_channels, in_channels, kernel_size];
    let num_els: usize = grad_weight_shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "conv1d_grad_weight")?;
    let command_buffer = device.command_buffer()?;

    let input_buf = BufferOffset {
        buffer: input.buffer(),
        offset_in_bytes: input_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let grad_output_buf = BufferOffset {
        buffer: grad_output.buffer(),
        offset_in_bytes: grad_output_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    // Metadata: [num_els, batch, in_channels, out_channels, input_length, output_length, kernel_size, stride, padding, dilation]
    let metadata = vec![
        num_els,
        batch,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        params.stride,
        params.padding,
        params.dilation,
    ];

    macro_rules! dispatch_conv1d_grad_weight {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "conv1d_grad_weight".to_string(),
                    })
                },
            }
        };
    }

    dispatch_conv1d_grad_weight!(conv1d_grad_weight);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

/// Convolution 2D gradient weight operation
pub(crate) fn conv2d_grad_weight_map(
    input: &MetalStorage,
    grad_output: &MetalStorage,
    input_layout: &Layout,
    grad_output_layout: &Layout,
    params: &ParamsConv2D,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_conv_grad_weight, utils::BufferOffset};

    let dtype = input.get_dtype();
    let device = input.get_hodu_device();

    if dtype != grad_output.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: grad_output.get_dtype(),
            op: "conv2d_grad_weight".to_string(),
        });
    }

    let input_shape = input_layout.get_shape();
    let grad_output_shape = grad_output_layout.get_shape();

    // Input: [batch, in_channels, height, width]
    // Grad Output: [batch, out_channels, output_height, output_width]
    // Grad Weight: [out_channels, in_channels, kernel_height, kernel_width]
    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];

    let out_channels = grad_output_shape[1];
    let output_height = grad_output_shape[2];
    let output_width = grad_output_shape[3];

    let kernel_height = params.kernel_height;
    let kernel_width = params.kernel_width;

    // Output is the grad_weight with shape [out_channels, in_channels, kernel_height, kernel_width]
    let grad_weight_shape = [out_channels, in_channels, kernel_height, kernel_width];
    let num_els: usize = grad_weight_shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "conv2d_grad_weight")?;
    let command_buffer = device.command_buffer()?;

    let input_buf = BufferOffset {
        buffer: input.buffer(),
        offset_in_bytes: input_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let grad_output_buf = BufferOffset {
        buffer: grad_output.buffer(),
        offset_in_bytes: grad_output_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    // Metadata: [num_els, batch, in_channels, out_channels, input_height, input_width,
    //            output_height, output_width, kernel_height, kernel_width, stride, padding, dilation]
    let metadata = vec![
        num_els,
        batch,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        params.stride,
        params.padding,
        params.dilation,
    ];

    macro_rules! dispatch_conv2d_grad_weight {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "conv2d_grad_weight".to_string(),
                    })
                },
            }
        };
    }

    dispatch_conv2d_grad_weight!(conv2d_grad_weight);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

/// Convolution 3D gradient weight operation
pub(crate) fn conv3d_grad_weight_map(
    input: &MetalStorage,
    grad_output: &MetalStorage,
    input_layout: &Layout,
    grad_output_layout: &Layout,
    params: &ParamsConv3D,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_conv_grad_weight, utils::BufferOffset};

    let dtype = input.get_dtype();
    let device = input.get_hodu_device();

    if dtype != grad_output.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: grad_output.get_dtype(),
            op: "conv3d_grad_weight".to_string(),
        });
    }

    let input_shape = input_layout.get_shape();
    let grad_output_shape = grad_output_layout.get_shape();

    // Input: [batch, in_channels, depth, height, width]
    // Grad Output: [batch, out_channels, output_depth, output_height, output_width]
    // Grad Weight: [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]
    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let input_depth = input_shape[2];
    let input_height = input_shape[3];
    let input_width = input_shape[4];

    let out_channels = grad_output_shape[1];
    let output_depth = grad_output_shape[2];
    let output_height = grad_output_shape[3];
    let output_width = grad_output_shape[4];

    let kernel_depth = params.kernel_depth;
    let kernel_height = params.kernel_height;
    let kernel_width = params.kernel_width;

    // Output is the grad_weight with shape [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]
    let grad_weight_shape = [out_channels, in_channels, kernel_depth, kernel_height, kernel_width];
    let num_els: usize = grad_weight_shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "conv3d_grad_weight")?;
    let command_buffer = device.command_buffer()?;

    let input_buf = BufferOffset {
        buffer: input.buffer(),
        offset_in_bytes: input_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let grad_output_buf = BufferOffset {
        buffer: grad_output.buffer(),
        offset_in_bytes: grad_output_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    // Metadata: [num_els, batch, in_channels, out_channels, input_depth, input_height, input_width,
    //            output_depth, output_height, output_width, kernel_depth, kernel_height, kernel_width,
    //            stride, padding, dilation]
    let metadata = vec![
        num_els,
        batch,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        params.stride,
        params.padding,
        params.dilation,
    ];

    macro_rules! dispatch_conv3d_grad_weight {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "conv3d_grad_weight".to_string(),
                    })
                },
            }
        };
    }

    dispatch_conv3d_grad_weight!(conv3d_grad_weight);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

/// Transpose convolution 1D gradient weight operation
pub(crate) fn conv_transpose1d_grad_weight_map(
    input: &MetalStorage,
    grad_output: &MetalStorage,
    input_layout: &Layout,
    grad_output_layout: &Layout,
    params: &ParamsConvTranspose1D,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_conv_grad_weight, utils::BufferOffset};

    let dtype = input.get_dtype();
    let device = input.get_hodu_device();

    if dtype != grad_output.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: grad_output.get_dtype(),
            op: "conv_transpose1d_grad_weight".to_string(),
        });
    }

    let input_shape = input_layout.get_shape();
    let grad_output_shape = grad_output_layout.get_shape();

    // Input: [batch, in_channels, input_length]
    // Grad Output: [batch, out_channels, output_length]
    // Grad Weight: [in_channels, out_channels, kernel_size]
    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let input_length = input_shape[2];

    let out_channels = grad_output_shape[1];
    let output_length = grad_output_shape[2];

    let kernel_size = params.kernel_size;

    // Output is the grad_weight with shape [in_channels, out_channels, kernel_size]
    let grad_weight_shape = [in_channels, out_channels, kernel_size];
    let num_els: usize = grad_weight_shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "conv_transpose1d_grad_weight")?;
    let command_buffer = device.command_buffer()?;

    let input_buf = BufferOffset {
        buffer: input.buffer(),
        offset_in_bytes: input_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let grad_output_buf = BufferOffset {
        buffer: grad_output.buffer(),
        offset_in_bytes: grad_output_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    // Metadata: [num_els, batch, in_channels, out_channels, input_length, output_length,
    //            kernel_size, padding, output_padding, stride, dilation]
    let metadata = vec![
        num_els,
        batch,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        params.padding,
        params.output_padding,
        params.stride,
        params.dilation,
    ];

    macro_rules! dispatch_conv_transpose1d_grad_weight {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "conv_transpose1d_grad_weight".to_string(),
                    })
                },
            }
        };
    }

    dispatch_conv_transpose1d_grad_weight!(conv_transpose1d_grad_weight);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

/// Transpose convolution 2D gradient weight operation
pub(crate) fn conv_transpose2d_grad_weight_map(
    input: &MetalStorage,
    grad_output: &MetalStorage,
    input_layout: &Layout,
    grad_output_layout: &Layout,
    params: &ParamsConvTranspose2D,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_conv_grad_weight, utils::BufferOffset};

    let dtype = input.get_dtype();
    let device = input.get_hodu_device();

    if dtype != grad_output.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: grad_output.get_dtype(),
            op: "conv_transpose2d_grad_weight".to_string(),
        });
    }

    let input_shape = input_layout.get_shape();
    let grad_output_shape = grad_output_layout.get_shape();

    // Input: [batch, in_channels, input_height, input_width]
    // Grad Output: [batch, out_channels, output_height, output_width]
    // Grad Weight: [in_channels, out_channels, kernel_height, kernel_width]
    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];

    let out_channels = grad_output_shape[1];
    let output_height = grad_output_shape[2];
    let output_width = grad_output_shape[3];

    let kernel_height = params.kernel_height;
    let kernel_width = params.kernel_width;

    // Output is the grad_weight with shape [in_channels, out_channels, kernel_height, kernel_width]
    let grad_weight_shape = [in_channels, out_channels, kernel_height, kernel_width];
    let num_els: usize = grad_weight_shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "conv_transpose2d_grad_weight")?;
    let command_buffer = device.command_buffer()?;

    let input_buf = BufferOffset {
        buffer: input.buffer(),
        offset_in_bytes: input_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let grad_output_buf = BufferOffset {
        buffer: grad_output.buffer(),
        offset_in_bytes: grad_output_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    // Metadata: [num_els, batch, in_channels, out_channels, input_height, input_width,
    //            output_height, output_width, kernel_height, kernel_width,
    //            padding, output_padding, stride, dilation]
    let metadata = vec![
        num_els,
        batch,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        params.padding,
        params.output_padding,
        params.stride,
        params.dilation,
    ];

    macro_rules! dispatch_conv_transpose2d_grad_weight {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "conv_transpose2d_grad_weight".to_string(),
                    })
                },
            }
        };
    }

    dispatch_conv_transpose2d_grad_weight!(conv_transpose2d_grad_weight);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

/// Transpose convolution 3D gradient weight operation
pub(crate) fn conv_transpose3d_grad_weight_map(
    input: &MetalStorage,
    grad_output: &MetalStorage,
    input_layout: &Layout,
    grad_output_layout: &Layout,
    params: &ParamsConvTranspose3D,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_conv_grad_weight, utils::BufferOffset};

    let dtype = input.get_dtype();
    let device = input.get_hodu_device();

    if dtype != grad_output.get_dtype() {
        return Err(HoduError::DTypeConflictInOp {
            left: dtype,
            right: grad_output.get_dtype(),
            op: "conv_transpose3d_grad_weight".to_string(),
        });
    }

    let input_shape = input_layout.get_shape();
    let grad_output_shape = grad_output_layout.get_shape();

    // Input: [batch, in_channels, input_depth, input_height, input_width]
    // Grad Output: [batch, out_channels, output_depth, output_height, output_width]
    // Grad Weight: [in_channels, out_channels, kernel_depth, kernel_height, kernel_width]
    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let input_depth = input_shape[2];
    let input_height = input_shape[3];
    let input_width = input_shape[4];

    let out_channels = grad_output_shape[1];
    let output_depth = grad_output_shape[2];
    let output_height = grad_output_shape[3];
    let output_width = grad_output_shape[4];

    let kernel_depth = params.kernel_depth;
    let kernel_height = params.kernel_height;
    let kernel_width = params.kernel_width;

    // Output is the grad_weight with shape [in_channels, out_channels, kernel_depth, kernel_height, kernel_width]
    let grad_weight_shape = [in_channels, out_channels, kernel_depth, kernel_height, kernel_width];
    let num_els: usize = grad_weight_shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "conv_transpose3d_grad_weight")?;
    let command_buffer = device.command_buffer()?;

    let input_buf = BufferOffset {
        buffer: input.buffer(),
        offset_in_bytes: input_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    let grad_output_buf = BufferOffset {
        buffer: grad_output.buffer(),
        offset_in_bytes: grad_output_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    // Metadata: [num_els, batch, in_channels, out_channels, input_depth, input_height, input_width,
    //            output_depth, output_height, output_width, kernel_depth, kernel_height, kernel_width,
    //            padding, output_padding, stride, dilation]
    let metadata = vec![
        num_els,
        batch,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        params.padding,
        params.output_padding,
        params.stride,
        params.dilation,
    ];

    macro_rules! dispatch_conv_transpose3d_grad_weight {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_conv_grad_weight(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        input_buf,
                        grad_output_buf,
                        &output,
                        &metadata,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: "conv_transpose3d_grad_weight".to_string(),
                    })
                },
            }
        };
    }

    dispatch_conv_transpose3d_grad_weight!(conv_transpose3d_grad_weight);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

/// Reduce window operation
pub(crate) fn reduce_window_map(
    input: &MetalStorage,
    input_layout: &Layout,
    window_shape: &[usize],
    strides: &[usize],
    padding: &[(usize, usize)],
    reduction: WindowReduction,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_reduce_window, utils::BufferOffset};

    let dtype = input.get_dtype();
    let device = input.get_hodu_device();

    let input_shape = input_layout.get_shape();
    let rank = input_shape.len();

    if window_shape.len() != rank || strides.len() != rank || padding.len() != rank {
        return Err(HoduError::InternalError(
            "window_shape, strides, and padding must have same rank as input".to_string(),
        ));
    }

    // Calculate output shape
    let mut output_shape = Vec::with_capacity(rank);
    for i in 0..rank {
        let padded_size = input_shape[i] + padding[i].0 + padding[i].1;
        let out_size = (padded_size - window_shape[i]) / strides[i] + 1;
        output_shape.push(out_size);
    }

    let num_els: usize = output_shape.iter().product();
    let output = device.new_buffer(num_els, dtype, "reduce_window")?;
    let command_buffer = device.command_buffer()?;

    let input_buf = BufferOffset {
        buffer: input.buffer(),
        offset_in_bytes: input_layout.get_offset() * dtype.get_size_in_bytes(),
    };

    // Flatten padding from [(usize, usize)] to [usize]
    let mut padding_flat = Vec::with_capacity(rank * 2);
    for &(before, after) in padding {
        padding_flat.push(before);
        padding_flat.push(after);
    }

    // Select kernel based on reduction type
    macro_rules! dispatch_reduce_window {
        ($kernel_mod:ident) => {
            match dtype {
                DType::BF16 => {
                    call_reduce_window(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::BF16,
                        input_shape,
                        input_buf,
                        input_layout.get_strides(),
                        input_layout.get_offset(),
                        window_shape,
                        strides,
                        &padding_flat,
                        &output_shape,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F16 => {
                    call_reduce_window(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F16,
                        input_shape,
                        input_buf,
                        input_layout.get_strides(),
                        input_layout.get_offset(),
                        window_shape,
                        strides,
                        &padding_flat,
                        &output_shape,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::F32 => {
                    call_reduce_window(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::F32,
                        input_shape,
                        input_buf,
                        input_layout.get_strides(),
                        input_layout.get_offset(),
                        window_shape,
                        strides,
                        &padding_flat,
                        &output_shape,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I8 => {
                    call_reduce_window(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I8,
                        input_shape,
                        input_buf,
                        input_layout.get_strides(),
                        input_layout.get_offset(),
                        window_shape,
                        strides,
                        &padding_flat,
                        &output_shape,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I16 => {
                    call_reduce_window(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I16,
                        input_shape,
                        input_buf,
                        input_layout.get_strides(),
                        input_layout.get_offset(),
                        window_shape,
                        strides,
                        &padding_flat,
                        &output_shape,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I32 => {
                    call_reduce_window(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I32,
                        input_shape,
                        input_buf,
                        input_layout.get_strides(),
                        input_layout.get_offset(),
                        window_shape,
                        strides,
                        &padding_flat,
                        &output_shape,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::I64 => {
                    call_reduce_window(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::I64,
                        input_shape,
                        input_buf,
                        input_layout.get_strides(),
                        input_layout.get_offset(),
                        window_shape,
                        strides,
                        &padding_flat,
                        &output_shape,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U8 => {
                    call_reduce_window(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U8,
                        input_shape,
                        input_buf,
                        input_layout.get_strides(),
                        input_layout.get_offset(),
                        window_shape,
                        strides,
                        &padding_flat,
                        &output_shape,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U16 => {
                    call_reduce_window(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U16,
                        input_shape,
                        input_buf,
                        input_layout.get_strides(),
                        input_layout.get_offset(),
                        window_shape,
                        strides,
                        &padding_flat,
                        &output_shape,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U32 => {
                    call_reduce_window(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U32,
                        input_shape,
                        input_buf,
                        input_layout.get_strides(),
                        input_layout.get_offset(),
                        window_shape,
                        strides,
                        &padding_flat,
                        &output_shape,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                DType::U64 => {
                    call_reduce_window(
                        device.device(),
                        &command_buffer,
                        device.kernels(),
                        hodu_metal_kernels::kernels::$kernel_mod::U64,
                        input_shape,
                        input_buf,
                        input_layout.get_strides(),
                        input_layout.get_offset(),
                        window_shape,
                        strides,
                        &padding_flat,
                        &output_shape,
                        &output,
                    )
                    .map_err(|e| HoduError::Metal(e.into()))?;
                },
                _ => {
                    return Err(HoduError::UnsupportedDType {
                        dtype,
                        op: format!("reduce_window_{}", reduction.to_string()),
                    })
                },
            }
        };
    }

    match reduction {
        WindowReduction::Max => {
            dispatch_reduce_window!(reduce_window_max);
        },
        WindowReduction::Min => {
            dispatch_reduce_window!(reduce_window_min);
        },
        WindowReduction::Sum => {
            dispatch_reduce_window!(reduce_window_sum);
        },
        WindowReduction::Mean => {
            dispatch_reduce_window!(reduce_window_mean);
        },
    }

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}

/// Convert storage to a different dtype using Metal cast kernels
pub(crate) fn to_dtype_map(
    input: &MetalStorage,
    input_layout: &Layout,
    target_dtype: DType,
) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{kernels::call_cast, utils::BufferOffset};

    let source_dtype = input.get_dtype();
    let device = input.get_hodu_device();

    // If already the target dtype, return clone
    if source_dtype == target_dtype {
        return Ok(input.clone());
    }

    let shape = input_layout.get_shape();
    let num_els: usize = shape.iter().product();

    let output = device.new_buffer(num_els, target_dtype, "to_dtype")?;
    let command_buffer = device.command_buffer()?;

    // Generate kernel name based on source and target dtypes
    let kernel_name = format!("cast_{}_to_{}", source_dtype, target_dtype);

    let input_buf = BufferOffset {
        buffer: input.buffer(),
        offset_in_bytes: input_layout.get_offset() * source_dtype.get_size_in_bytes(),
    };

    call_cast(
        device.device(),
        &command_buffer,
        device.kernels(),
        Box::leak(kernel_name.into_boxed_str()),
        shape,
        input_buf,
        input_layout.get_strides(),
        input_layout.get_offset(),
        &output,
    )
    .map_err(|e| HoduError::Metal(e.into()))?;

    Ok(MetalStorage::new(output, device.clone(), num_els, target_dtype))
}

/// Make storage contiguous using Metal contiguous kernel
pub(crate) fn contiguous_map(input: &MetalStorage, layout: &Layout) -> HoduResult<MetalStorage> {
    use hodu_metal_kernels::{
        kernels::{call_contiguous, Kernel},
        utils::BufferOffset,
    };

    let device = input.get_hodu_device();
    let dtype = input.get_dtype();

    // If already contiguous, return clone
    if layout.is_contiguous() {
        return Ok(input.clone());
    }

    let shape = layout.get_shape();
    let num_els: usize = shape.iter().product();

    let output = device.new_buffer(num_els, dtype, "contiguous")?;
    let command_buffer = device.command_buffer()?;

    let input_buf = BufferOffset {
        buffer: input.buffer(),
        offset_in_bytes: layout.get_offset() * dtype.get_size_in_bytes(),
    };

    // Generate kernel name based on dtype
    let kernel_name = format!("contiguous_{}", dtype);

    // Use contiguous kernel to convert strided layout to contiguous
    call_contiguous(
        device.device(),
        &command_buffer,
        device.kernels(),
        Kernel(Box::leak(kernel_name.into_boxed_str())),
        shape,
        input_buf,
        layout.get_strides(),
        layout.get_offset(),
        &output,
    )
    .map_err(|e| HoduError::Metal(e.into()))?;

    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}
