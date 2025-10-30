use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    op::conv::ParamsConv3D,
    types::{dtype::DType, layout::Layout},
};

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
                        op: "conv3d".to_string(),
                    })
                },
            }
        };
    }

    dispatch_conv3d!(conv3d);
    Ok(MetalStorage::new(output, device.clone(), num_els, dtype))
}
