use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    types::{dtype::DType, layout::Layout},
};

/// Common dtype dispatch macro for binary operations.
/// Reduces code duplication across binary_map and binary_logical_map.
macro_rules! dispatch_binary_dtype {
    ($op:ident, $dtype:expr, $device:expr, $command_buffer:expr, $kernels:expr, $output_shape:expr, $lhs:expr, $lhs_strides:expr, $lhs_offset:expr, $rhs:expr, $rhs_strides:expr, $rhs_offset:expr, $output:expr, $kernel_name:expr) => {
        match $dtype {
            DType::BF16 => {
                hodu_metal_kernels::kernels::call_binary(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$op::BF16,
                    $output_shape,
                    $lhs,
                    $lhs_strides,
                    $lhs_offset,
                    $rhs,
                    $rhs_strides,
                    $rhs_offset,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::F16 => {
                hodu_metal_kernels::kernels::call_binary(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$op::F16,
                    $output_shape,
                    $lhs,
                    $lhs_strides,
                    $lhs_offset,
                    $rhs,
                    $rhs_strides,
                    $rhs_offset,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::F32 => {
                hodu_metal_kernels::kernels::call_binary(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$op::F32,
                    $output_shape,
                    $lhs,
                    $lhs_strides,
                    $lhs_offset,
                    $rhs,
                    $rhs_strides,
                    $rhs_offset,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            #[cfg(feature = "u8")]
            DType::U8 => {
                hodu_metal_kernels::kernels::call_binary(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$op::U8,
                    $output_shape,
                    $lhs,
                    $lhs_strides,
                    $lhs_offset,
                    $rhs,
                    $rhs_strides,
                    $rhs_offset,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::U16 => {
                hodu_metal_kernels::kernels::call_binary(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$op::U16,
                    $output_shape,
                    $lhs,
                    $lhs_strides,
                    $lhs_offset,
                    $rhs,
                    $rhs_strides,
                    $rhs_offset,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            #[cfg(feature = "u32")]
            DType::U32 => {
                hodu_metal_kernels::kernels::call_binary(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$op::U32,
                    $output_shape,
                    $lhs,
                    $lhs_strides,
                    $lhs_offset,
                    $rhs,
                    $rhs_strides,
                    $rhs_offset,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            #[cfg(feature = "u64")]
            DType::U64 => {
                hodu_metal_kernels::kernels::call_binary(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$op::U64,
                    $output_shape,
                    $lhs,
                    $lhs_strides,
                    $lhs_offset,
                    $rhs,
                    $rhs_strides,
                    $rhs_offset,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::I8 => {
                hodu_metal_kernels::kernels::call_binary(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$op::I8,
                    $output_shape,
                    $lhs,
                    $lhs_strides,
                    $lhs_offset,
                    $rhs,
                    $rhs_strides,
                    $rhs_offset,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            #[cfg(feature = "i16")]
            DType::I16 => {
                hodu_metal_kernels::kernels::call_binary(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$op::I16,
                    $output_shape,
                    $lhs,
                    $lhs_strides,
                    $lhs_offset,
                    $rhs,
                    $rhs_strides,
                    $rhs_offset,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            DType::I32 => {
                hodu_metal_kernels::kernels::call_binary(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$op::I32,
                    $output_shape,
                    $lhs,
                    $lhs_strides,
                    $lhs_offset,
                    $rhs,
                    $rhs_strides,
                    $rhs_offset,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            #[cfg(feature = "i64")]
            DType::I64 => {
                hodu_metal_kernels::kernels::call_binary(
                    $device,
                    $command_buffer,
                    $kernels,
                    hodu_metal_kernels::kernels::$op::I64,
                    $output_shape,
                    $lhs,
                    $lhs_strides,
                    $lhs_offset,
                    $rhs,
                    $rhs_strides,
                    $rhs_offset,
                    $output,
                )
                .map_err(|e| HoduError::Metal(e.into()))?;
            },
            _ => {
                return Err(HoduError::UnsupportedDType {
                    dtype: $dtype,
                    op: format!("binary operation '{}'", $kernel_name),
                });
            },
        }
    };
}

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

    match kernel_name {
        "add" => dispatch_binary_dtype!(
            add,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
        "sub" => dispatch_binary_dtype!(
            sub,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
        "mul" => dispatch_binary_dtype!(
            mul,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
        "div" => dispatch_binary_dtype!(
            div,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
        "pow" => dispatch_binary_dtype!(
            pow,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
        "maximum" => dispatch_binary_dtype!(
            maximum,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
        "minimum" => dispatch_binary_dtype!(
            minimum,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
        "eq" => dispatch_binary_dtype!(
            eq,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
        "ne" => dispatch_binary_dtype!(
            ne,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
        "lt" => dispatch_binary_dtype!(
            lt,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
        "le" => dispatch_binary_dtype!(
            le,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
        "gt" => dispatch_binary_dtype!(
            gt,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
        "ge" => dispatch_binary_dtype!(
            ge,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
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

    match kernel_name {
        "logical_and" => dispatch_binary_dtype!(
            logical_and,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
        "logical_or" => dispatch_binary_dtype!(
            logical_or,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
        "logical_xor" => dispatch_binary_dtype!(
            logical_xor,
            dtype,
            device.device(),
            &command_buffer,
            device.kernels(),
            &output_shape,
            lhs,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            &output,
            kernel_name
        ),
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype,
                op: format!("unknown binary logical operation '{}'", kernel_name),
            });
        },
    }

    Ok(MetalStorage::new(output, device.clone(), output_size, output_dtype))
}
