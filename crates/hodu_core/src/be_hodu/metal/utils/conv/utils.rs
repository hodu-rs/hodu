use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    op::conv::{
        ParamsConv1D, ParamsConv2D, ParamsConv3D, ParamsConvTranspose1D, ParamsConvTranspose2D, ParamsConvTranspose3D,
    },
    types::{dtype::DType, layout::Layout},
};

/// Convolution 1D gradient weight operation
pub fn conv1d_grad_weight_map(
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
                #[cfg(feature = "i16")]
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
                #[cfg(feature = "i64")]
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
                #[cfg(feature = "u8")]
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
                #[cfg(feature = "u32")]
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
                #[cfg(feature = "u64")]
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
pub fn conv2d_grad_weight_map(
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
                #[cfg(feature = "i16")]
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
                #[cfg(feature = "i64")]
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
                #[cfg(feature = "u8")]
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
                #[cfg(feature = "u32")]
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
                #[cfg(feature = "u64")]
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
pub fn conv3d_grad_weight_map(
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
                #[cfg(feature = "i16")]
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
                #[cfg(feature = "i64")]
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
                #[cfg(feature = "u8")]
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
                #[cfg(feature = "u32")]
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
                #[cfg(feature = "u64")]
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
pub fn conv_transpose1d_grad_weight_map(
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
                #[cfg(feature = "i16")]
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
                #[cfg(feature = "i64")]
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
                #[cfg(feature = "u8")]
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
                #[cfg(feature = "u32")]
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
                #[cfg(feature = "u64")]
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
pub fn conv_transpose2d_grad_weight_map(
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
                #[cfg(feature = "i16")]
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
                #[cfg(feature = "i64")]
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
                #[cfg(feature = "u8")]
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
                #[cfg(feature = "u32")]
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
                #[cfg(feature = "u64")]
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
pub fn conv_transpose3d_grad_weight_map(
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
                #[cfg(feature = "i16")]
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
                #[cfg(feature = "i64")]
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
                #[cfg(feature = "u8")]
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
                #[cfg(feature = "u32")]
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
                #[cfg(feature = "u64")]
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
