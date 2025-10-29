use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    op::conv::ParamsConv1D,
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
