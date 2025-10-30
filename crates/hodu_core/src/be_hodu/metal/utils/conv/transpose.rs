use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    op::conv::{ParamsConvTranspose1D, ParamsConvTranspose2D, ParamsConvTranspose3D},
    types::{dtype::DType, layout::Layout},
};

/// Transpose convolution 1D operation
pub fn conv_transpose1d_map(
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
                #[cfg(feature = "i16")]
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
                #[cfg(feature = "i64")]
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
                #[cfg(feature = "u8")]
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
                #[cfg(feature = "u32")]
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
                #[cfg(feature = "u64")]
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
pub fn conv_transpose2d_map(
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
                #[cfg(feature = "i16")]
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
                #[cfg(feature = "i64")]
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
                #[cfg(feature = "u8")]
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
                #[cfg(feature = "u32")]
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
                #[cfg(feature = "u64")]
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
pub fn conv_transpose3d_map(
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
                #[cfg(feature = "i16")]
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
                #[cfg(feature = "i64")]
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
                #[cfg(feature = "u8")]
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
                #[cfg(feature = "u32")]
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
                #[cfg(feature = "u64")]
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
