use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    types::{Layout, Shape},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_ops_index_select(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    dim: usize,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    match op {
        Op::Indexing(_) => (),
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_index_select expects indexing op".to_string(),
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
    output_shape_vec[dim] = num_indices;
    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();

    let metadata = crate::op_metadatas::index_select_metadata(input_layout, dim, num_indices, num_els);

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els, dtype, "index_select_output")?;

    // Get kernel name
    let kernel_name = format!("hodu_metal_index_select_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets
    let input_offset_buf = BufferOffset::zero_offset(input_storage.buffer());
    let indices_offset_buf = BufferOffset::zero_offset(indices_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_index_select(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset_buf,
        indices_offset_buf,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, dtype))
}

#[allow(clippy::too_many_arguments)]
pub fn call_ops_index_put(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    values_storage: &MetalStorage,
    values_layout: &Layout,
    dim: usize,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    match op {
        Op::Indexing(_) => (),
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_index_put expects indexing op".to_string(),
            ))
        },
    }

    let input_shape = input_layout.shape();
    let indices_shape = indices_layout.shape();
    let num_els = input_shape.size();
    let num_indices = indices_shape.size();

    let metadata = crate::op_metadatas::index_put_metadata(input_layout, values_layout, dim, num_indices, num_els);

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer (copy of input)
    let output_buffer = device.new_buffer(num_els, dtype, "index_put_output")?;

    // First copy input to output
    let command_buffer = device.command_buffer()?;
    let blit = command_buffer.blit_command_encoder();
    blit.copy_from_buffer(
        input_storage.buffer(),
        0,
        &output_buffer,
        0,
        num_els * dtype.size_in_bytes(),
    );
    blit.end_encoding();

    // Get kernel name
    let kernel_name = format!("hodu_metal_index_put_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets
    let input_offset_buf = BufferOffset::zero_offset(input_storage.buffer());
    let indices_offset_buf = BufferOffset::zero_offset(indices_storage.buffer());
    let values_offset_buf = BufferOffset::zero_offset(values_storage.buffer());

    // Call kernel
    kernels::call_ops_index_put(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset_buf,
        indices_offset_buf,
        values_offset_buf,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, dtype))
}

pub fn call_ops_gather(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    dim: usize,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    match op {
        Op::Indexing(_) => (),
        _ => {
            return Err(HoduError::BackendError(
                "Lcall_ops_gatherE expects LindexingE op".to_string(),
            ))
        },
    }

    let indices_shape = indices_layout.shape();
    let output_shape = indices_shape.clone();
    let num_els = output_shape.size();

    let metadata = crate::op_metadatas::gather_metadata(input_layout, indices_layout, dim, num_els);

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els, dtype, "gather_output")?;

    // Get kernel name
    let kernel_name = format!("hodu_metal_gather_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets
    let input_offset_buf = BufferOffset::zero_offset(input_storage.buffer());
    let indices_offset_buf = BufferOffset::zero_offset(indices_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_gather(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset_buf,
        indices_offset_buf,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, dtype))
}

#[allow(clippy::too_many_arguments)]
pub fn call_ops_scatter(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    src_storage: &MetalStorage,
    src_layout: &Layout,
    dim: usize,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    match op {
        Op::Indexing(_) => (),
        _ => {
            return Err(HoduError::BackendError(
                "Lcall_ops_scatterE expects LindexingE op".to_string(),
            ))
        },
    }

    let input_shape = input_layout.shape();
    let num_els = input_shape.size();

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer (copy of input)
    let output_buffer = device.new_buffer(num_els, dtype, "scatter_output")?;

    // First copy input to output
    let command_buffer = device.command_buffer()?;
    let blit = command_buffer.blit_command_encoder();
    blit.copy_from_buffer(
        input_storage.buffer(),
        0,
        &output_buffer,
        0,
        num_els * dtype.size_in_bytes(),
    );
    blit.end_encoding();

    // Get kernel name
    let kernel_name = format!("hodu_metal_scatter_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Generate metadata using centralized function
    let metadata = crate::op_metadatas::scatter_metadata(input_layout, indices_layout, src_layout, dim);

    // Create buffer offsets
    let input_offset_buf = BufferOffset::zero_offset(input_storage.buffer());
    let indices_offset_buf = BufferOffset::zero_offset(indices_storage.buffer());
    let src_offset_buf = BufferOffset::zero_offset(src_storage.buffer());

    // Call kernel
    kernels::call_ops_scatter(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset_buf,
        indices_offset_buf,
        src_offset_buf,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, dtype))
}
