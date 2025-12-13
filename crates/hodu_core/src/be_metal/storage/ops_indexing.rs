use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::{IndexingOp, Op},
    types::{DType, Layout, Shape},
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

pub fn call_ops_onehot(
    indices_storage: &MetalStorage,
    indices_layout: &Layout,
    num_classes: usize,
    axis: usize,
    output_dtype: DType,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    match op {
        Op::Indexing(IndexingOp::Onehot) => (),
        _ => return Err(HoduError::BackendError("call_ops_onehot expects Onehot op".to_string())),
    }

    let input_shape = indices_layout.shape();
    let num_dims_in = input_shape.ndim();
    let num_input_els = input_shape.size();

    // Compute output shape: insert num_classes at axis position
    let mut output_shape_vec = Vec::with_capacity(num_dims_in + 1);
    for (i, &dim) in input_shape.dims().iter().enumerate() {
        if i == axis {
            output_shape_vec.push(num_classes);
        }
        output_shape_vec.push(dim);
    }
    // If axis == num_dims_in (last position)
    if axis == num_dims_in {
        output_shape_vec.push(num_classes);
    }

    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();
    let num_dims_out = output_shape.ndim();

    let device = indices_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els, output_dtype, "onehot_output")?;

    // Get kernel name
    let kernel_name = format!("hodu_metal_onehot_{}", output_dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Generate metadata
    // - metadata[0]: num_els (total number of output elements)
    // - metadata[1]: num_input_els (total number of input indices)
    // - metadata[2]: num_classes (depth of one-hot dimension)
    // - metadata[3]: axis (dimension for one-hot encoding)
    // - metadata[4]: num_dims_out (number of output dimensions)
    // - metadata[5..5+num_dims_out]: output_shape
    let mut metadata = Vec::with_capacity(5 + num_dims_out);
    metadata.push(num_els);
    metadata.push(num_input_els);
    metadata.push(num_classes);
    metadata.push(axis);
    metadata.push(num_dims_out);
    metadata.extend_from_slice(output_shape.dims());

    // Create buffer offsets
    let indices_offset_buf = BufferOffset::zero_offset(indices_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_onehot(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        indices_offset_buf,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, output_dtype))
}

pub fn call_nonzero(input_storage: &MetalStorage, input_layout: &Layout) -> HoduResult<(MetalStorage, usize)> {
    let dtype = input_storage.dtype();
    let shape = input_layout.shape();
    let ndim = shape.ndim();
    let num_els = shape.size();

    let device = input_storage.backend_device();

    // Build metadata
    let mut metadata = Vec::with_capacity(2 + 2 * ndim + 1);
    metadata.push(num_els);
    metadata.push(ndim);
    metadata.extend_from_slice(shape.dims());
    metadata.extend_from_slice(input_layout.strides());
    metadata.push(input_layout.offset());

    // Create atomic counter buffer (single u32)
    let count_buffer = device.new_buffer(1, DType::U32, "nonzero_count")?;

    // First pass: count non-zero elements
    let count_kernel_name = format!("hodu_metal_nonzero_count_{}", dtype);
    let count_kernel_name_static = crate::cache::kernel::get_kernel_name(count_kernel_name);
    let count_kernel = kernels::Kernel(count_kernel_name_static);

    let input_offset = BufferOffset::zero_offset(input_storage.buffer());
    let command_buffer = device.command_buffer()?;

    kernels::call_nonzero_count(
        count_kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &count_buffer,
        &metadata,
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read count from buffer
    let count_ptr = count_buffer.contents() as *const u32;
    let count = unsafe { *count_ptr } as usize;

    // Handle empty case
    if count == 0 {
        let output_buffer = device.new_buffer(0, DType::I32, "nonzero_output")?;
        return Ok((MetalStorage::new(output_buffer, device.clone(), 0, DType::I32), 0));
    }

    // Allocate output buffer for [count, ndim] indices
    let output_size = count * ndim;
    let output_buffer = device.new_buffer(output_size, DType::I32, "nonzero_output")?;

    // Create a new counter buffer for fill (reset to 0)
    let counter_buffer = device.new_buffer(1, DType::U32, "nonzero_counter")?;

    // Second pass: fill indices
    let fill_kernel_name = format!("hodu_metal_nonzero_fill_{}", dtype);
    let fill_kernel_name_static = crate::cache::kernel::get_kernel_name(fill_kernel_name);
    let fill_kernel = kernels::Kernel(fill_kernel_name_static);

    let input_offset = BufferOffset::zero_offset(input_storage.buffer());
    let command_buffer = device.command_buffer()?;

    kernels::call_nonzero_fill(
        fill_kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &output_buffer,
        &counter_buffer,
        &metadata,
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok((
        MetalStorage::new(output_buffer, device.clone(), output_size, DType::I32),
        count,
    ))
}

/// Unique operation - uses GPU kernels for sorting and finding unique elements
pub fn call_unique(
    input_storage: &MetalStorage,
    input_layout: &Layout,
) -> HoduResult<(MetalStorage, MetalStorage, MetalStorage, usize)> {
    let dtype = input_storage.dtype();
    let shape = input_layout.shape();
    let num_els = shape.size();

    if num_els == 0 {
        let device = input_storage.backend_device();
        let empty_values = device.new_buffer(0, dtype, "unique_values")?;
        let empty_inverse = device.new_buffer(0, DType::I32, "unique_inverse")?;
        let empty_counts = device.new_buffer(0, DType::I32, "unique_counts")?;
        return Ok((
            MetalStorage::new(empty_values, device.clone(), 0, dtype),
            MetalStorage::new(empty_inverse, device.clone(), 0, DType::I32),
            MetalStorage::new(empty_counts, device.clone(), 0, DType::I32),
            0,
        ));
    }

    let device = input_storage.backend_device();

    // Compute next power of 2 for bitonic sort
    let padded_size = num_els.next_power_of_two();

    // Metadata: [num_els, offset, padded_size]
    let metadata = vec![num_els, input_layout.offset(), padded_size];

    // Step 1: Copy input to sorted_values and initialize sorted_indices (with padding)
    let sorted_values = device.new_buffer(padded_size, dtype, "unique_sorted_values")?;
    let sorted_indices = device.new_buffer(padded_size, DType::I32, "unique_sorted_indices")?;

    let sort_kernel_name = format!("hodu_metal_unique_sort_{}", dtype);
    let sort_kernel_name_static = crate::cache::kernel::get_kernel_name(sort_kernel_name);
    let sort_kernel = kernels::Kernel(sort_kernel_name_static);

    let input_offset = BufferOffset::zero_offset(input_storage.buffer());
    let command_buffer = device.command_buffer()?;

    kernels::call_unique_sort(
        sort_kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &sorted_values,
        &sorted_indices,
        &metadata,
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Step 2: Bitonic sort
    let bitonic_kernel_name = format!("hodu_metal_unique_bitonic_step_{}", dtype);
    let bitonic_kernel_name_static = crate::cache::kernel::get_kernel_name(bitonic_kernel_name);
    let bitonic_kernel = kernels::Kernel(bitonic_kernel_name_static);

    let mut k = 2;
    while k <= padded_size {
        let mut j = k / 2;
        while j >= 1 {
            // Bitonic metadata: [num_els, offset, padded_size, k, j]
            let bitonic_metadata = vec![num_els, input_layout.offset(), padded_size, k, j];
            let command_buffer = device.command_buffer()?;

            kernels::call_unique_bitonic_step(
                bitonic_kernel,
                device.kernels(),
                device.device(),
                &command_buffer,
                &sorted_values,
                &sorted_indices,
                &bitonic_metadata,
            )?;

            command_buffer.commit();
            command_buffer.wait_until_completed();

            j /= 2;
        }
        k *= 2;
    }

    // Step 3: Count unique elements
    let count_buffer = device.new_buffer(1, DType::U32, "unique_count")?;

    let count_kernel_name = format!("hodu_metal_unique_count_{}", dtype);
    let count_kernel_name_static = crate::cache::kernel::get_kernel_name(count_kernel_name);
    let count_kernel = kernels::Kernel(count_kernel_name_static);

    let command_buffer = device.command_buffer()?;
    kernels::call_unique_count(
        count_kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        &sorted_values,
        &count_buffer,
        &metadata,
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read unique count
    let count_ptr = count_buffer.contents() as *const u32;
    let unique_count = unsafe { *count_ptr } as usize;

    if unique_count == 0 {
        let empty_values = device.new_buffer(0, dtype, "unique_values")?;
        let empty_inverse = device.new_buffer(0, DType::I32, "unique_inverse")?;
        let empty_counts = device.new_buffer(0, DType::I32, "unique_counts")?;
        return Ok((
            MetalStorage::new(empty_values, device.clone(), 0, dtype),
            MetalStorage::new(empty_inverse, device.clone(), 0, DType::I32),
            MetalStorage::new(empty_counts, device.clone(), 0, DType::I32),
            0,
        ));
    }

    // Step 4: Mark unique boundaries
    let marks = device.new_buffer(num_els, DType::U32, "unique_marks")?;

    let mark_kernel_name = format!("hodu_metal_unique_mark_{}", dtype);
    let mark_kernel_name_static = crate::cache::kernel::get_kernel_name(mark_kernel_name);
    let mark_kernel = kernels::Kernel(mark_kernel_name_static);

    let command_buffer = device.command_buffer()?;
    kernels::call_unique_mark(
        mark_kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        &sorted_values,
        &marks,
        &metadata,
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Step 5: Prefix sum to get unique indices
    let unique_idx = device.new_buffer(num_els, DType::I32, "unique_idx")?;

    let command_buffer = device.command_buffer()?;
    kernels::call_unique_prefix_sum(
        device.kernels(),
        device.device(),
        &command_buffer,
        &marks,
        &unique_idx,
        &metadata,
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Step 6: Build output arrays
    let values_buffer = device.new_buffer(unique_count, dtype, "unique_values")?;
    let inverse_buffer = device.new_buffer(num_els, DType::I32, "unique_inverse")?;
    let counts_buffer = device.new_buffer(unique_count, DType::I32, "unique_counts")?;

    // Initialize counts to 0
    {
        let zeros = vec![0i32; unique_count];
        let command_buffer = device.command_buffer()?;
        let blit = command_buffer.blit_command_encoder();
        let temp_buffer = device.new_buffer_with_data(&zeros)?;
        blit.copy_from_buffer(
            &temp_buffer,
            0,
            &counts_buffer,
            0,
            unique_count * DType::I32.size_in_bytes(),
        );
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    let build_kernel_name = format!("hodu_metal_unique_build_{}", dtype);
    let build_kernel_name_static = crate::cache::kernel::get_kernel_name(build_kernel_name);
    let build_kernel = kernels::Kernel(build_kernel_name_static);

    let command_buffer = device.command_buffer()?;
    kernels::call_unique_build(
        build_kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        &sorted_values,
        &sorted_indices,
        &marks,
        &unique_idx,
        &values_buffer,
        &inverse_buffer,
        &counts_buffer,
        &metadata,
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok((
        MetalStorage::new(values_buffer, device.clone(), unique_count, dtype),
        MetalStorage::new(inverse_buffer, device.clone(), num_els, DType::I32),
        MetalStorage::new(counts_buffer, device.clone(), unique_count, DType::I32),
        unique_count,
    ))
}
