use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    types::{Layout, Shape},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_index_select(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    dim: u32,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    match op {
        Op::Indexing(_) => (),
        _ => {
            return Err(HoduError::InternalError(
                "call_index_select expects indexing op".to_string(),
            ))
        },
    }

    let input_shape = input_layout.shape();
    let indices_shape = indices_layout.shape();
    let input_ndim = input_shape.ndim();

    // Validate dim
    if dim >= input_ndim {
        return Err(HoduError::InvalidAxis {
            axis: dim as i32,
            ndim: input_ndim,
        });
    }

    // Compute output shape
    let mut output_shape_vec = input_shape.dims().to_vec();
    let num_indices = indices_shape.size();
    output_shape_vec[dim as usize] = num_indices;
    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els as usize, dtype, "index_select_output")?;

    // Get kernel name
    let kernel_name = format!("index_select_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Convert to usize for kernel call
    let shape_usize: Vec<usize> = input_shape.dims().iter().map(|&d| d as usize).collect();
    let input_strides_usize: Vec<usize> = input_layout.strides().iter().map(|&s| s as usize).collect();

    // Create buffer offsets
    let input_offset_buf = BufferOffset::zero_offset(input_storage.buffer());
    let indices_offset_buf = BufferOffset::zero_offset(indices_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_index_select(
        device.device(),
        &command_buffer,
        device.kernels(),
        kernel,
        &shape_usize,
        input_offset_buf,
        &input_strides_usize,
        input_layout.offset() as usize,
        indices_offset_buf,
        dim as usize,
        num_indices as usize,
        &output_buffer,
    )?;

    Ok(MetalStorage::new(
        output_buffer,
        device.clone(),
        num_els as usize,
        dtype,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn call_index_put(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    values_storage: &MetalStorage,
    values_layout: &Layout,
    dim: u32,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    match op {
        Op::Indexing(_) => (),
        _ => {
            return Err(HoduError::InternalError(
                "call_index_put expects indexing op".to_string(),
            ))
        },
    }

    let input_shape = input_layout.shape();
    let indices_shape = indices_layout.shape();
    let num_els = input_shape.size();
    let num_indices = indices_shape.size();

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer (copy of input)
    let output_buffer = device.new_buffer(num_els as usize, dtype, "index_put_output")?;

    // First copy input to output
    let command_buffer = device.command_buffer()?;
    let blit = command_buffer.blit_command_encoder();
    blit.copy_from_buffer(
        input_storage.buffer(),
        0,
        &output_buffer,
        0,
        (num_els as usize) * dtype.get_size_in_bytes(),
    );
    blit.end_encoding();

    // Get kernel name
    let kernel_name = format!("index_put_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Convert to usize for kernel call
    let input_shape_usize: Vec<usize> = input_shape.dims().iter().map(|&d| d as usize).collect();
    let input_strides_usize: Vec<usize> = input_layout.strides().iter().map(|&s| s as usize).collect();
    let values_strides_usize: Vec<usize> = values_layout.strides().iter().map(|&s| s as usize).collect();

    // Create buffer offsets
    let input_offset_buf = BufferOffset::zero_offset(input_storage.buffer());
    let indices_offset_buf = BufferOffset::zero_offset(indices_storage.buffer());
    let values_offset_buf = BufferOffset::zero_offset(values_storage.buffer());

    // Call kernel
    kernels::call_index_put(
        device.device(),
        &command_buffer,
        device.kernels(),
        kernel,
        &input_shape_usize,
        input_offset_buf,
        &input_strides_usize,
        input_layout.offset() as usize,
        indices_offset_buf,
        values_offset_buf,
        &values_strides_usize,
        values_layout.offset() as usize,
        dim as usize,
        num_indices as usize,
        &output_buffer,
    )?;

    Ok(MetalStorage::new(
        output_buffer,
        device.clone(),
        num_els as usize,
        dtype,
    ))
}

pub fn call_gather(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    dim: u32,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    match op {
        Op::Indexing(_) => (),
        _ => return Err(HoduError::InternalError("call_gather expects indexing op".to_string())),
    }

    let input_shape = input_layout.shape();
    let indices_shape = indices_layout.shape();
    let output_shape = indices_shape.clone();
    let num_els = output_shape.size();

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els as usize, dtype, "gather_output")?;

    // Get kernel name
    let kernel_name = format!("gather_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Convert to usize for kernel call
    let input_shape_usize: Vec<usize> = input_shape.dims().iter().map(|&d| d as usize).collect();
    let input_strides_usize: Vec<usize> = input_layout.strides().iter().map(|&s| s as usize).collect();
    let indices_strides_usize: Vec<usize> = indices_layout.strides().iter().map(|&s| s as usize).collect();

    // Create buffer offsets
    let input_offset_buf = BufferOffset::zero_offset(input_storage.buffer());
    let indices_offset_buf = BufferOffset::zero_offset(indices_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_gather(
        device.device(),
        &command_buffer,
        device.kernels(),
        kernel,
        &input_shape_usize,
        input_offset_buf,
        &input_strides_usize,
        input_layout.offset() as usize,
        indices_offset_buf,
        &indices_strides_usize,
        indices_layout.offset() as usize,
        dim as usize,
        &output_buffer,
    )?;

    Ok(MetalStorage::new(
        output_buffer,
        device.clone(),
        num_els as usize,
        dtype,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn call_scatter(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    src_storage: &MetalStorage,
    src_layout: &Layout,
    dim: u32,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    match op {
        Op::Indexing(_) => (),
        _ => return Err(HoduError::InternalError("call_scatter expects indexing op".to_string())),
    }

    let input_shape = input_layout.shape();
    let src_shape = src_layout.shape();
    let num_els = input_shape.size();

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer (copy of input)
    let output_buffer = device.new_buffer(num_els as usize, dtype, "scatter_output")?;

    // First copy input to output
    let command_buffer = device.command_buffer()?;
    let blit = command_buffer.blit_command_encoder();
    blit.copy_from_buffer(
        input_storage.buffer(),
        0,
        &output_buffer,
        0,
        (num_els as usize) * dtype.get_size_in_bytes(),
    );
    blit.end_encoding();

    // Get kernel name
    let kernel_name = format!("scatter_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Convert to usize for kernel call
    let input_shape_usize: Vec<usize> = input_shape.dims().iter().map(|&d| d as usize).collect();
    let input_strides_usize: Vec<usize> = input_layout.strides().iter().map(|&s| s as usize).collect();
    let src_shape_usize: Vec<usize> = src_shape.dims().iter().map(|&d| d as usize).collect();
    let src_strides_usize: Vec<usize> = src_layout.strides().iter().map(|&s| s as usize).collect();
    let indices_strides_usize: Vec<usize> = indices_layout.strides().iter().map(|&s| s as usize).collect();

    // Create buffer offsets
    let input_offset_buf = BufferOffset::zero_offset(input_storage.buffer());
    let indices_offset_buf = BufferOffset::zero_offset(indices_storage.buffer());
    let src_offset_buf = BufferOffset::zero_offset(src_storage.buffer());

    // Call kernel
    kernels::call_scatter(
        device.device(),
        &command_buffer,
        device.kernels(),
        kernel,
        &input_shape_usize,
        input_offset_buf,
        &input_strides_usize,
        input_layout.offset() as usize,
        indices_offset_buf,
        &indices_strides_usize,
        indices_layout.offset() as usize,
        src_offset_buf,
        &src_shape_usize,
        &src_strides_usize,
        src_layout.offset() as usize,
        dim as usize,
        &output_buffer,
    )?;

    Ok(MetalStorage::new(
        output_buffer,
        device.clone(),
        num_els as usize,
        dtype,
    ))
}
