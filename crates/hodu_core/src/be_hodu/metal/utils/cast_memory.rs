use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorageT},
    error::{HoduError, HoduResult},
    types::{dtype::DType, layout::Layout},
};

/// Convert storage to a different dtype using Metal cast kernels
pub fn to_dtype_map(input: &MetalStorage, input_layout: &Layout, target_dtype: DType) -> HoduResult<MetalStorage> {
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
pub fn contiguous_map(input: &MetalStorage, layout: &Layout) -> HoduResult<MetalStorage> {
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
