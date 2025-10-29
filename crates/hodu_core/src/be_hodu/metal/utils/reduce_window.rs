use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    op::window_reduction::WindowReduction,
    types::{dtype::DType, layout::Layout},
};

/// Reduce window operation
pub fn reduce_window_map(
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
