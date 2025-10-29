use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    op::conv::{
        ParamsConv1D, ParamsConv2D, ParamsConv3D, ParamsConvTranspose1D, ParamsConvTranspose2D, ParamsConvTranspose3D,
    },
    types::{dtype::DType, layout::Layout},
};

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
