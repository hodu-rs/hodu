use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    types::{Layout, Shape},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

#[allow(clippy::needless_range_loop)]
pub fn call_ops_concat(
    first: &MetalStorage,
    others: &[&MetalStorage],
    layouts: &[&Layout],
    dim: usize,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Collect all storages
    let mut storages: Vec<&MetalStorage> = vec![first];
    storages.extend(others.iter().copied());

    // Validate op
    match op {
        Op::Concat(_) => (),
        _ => {
            return Err(HoduError::BackendError(
                "Lcall_ops_concatE expects LconcatE op".to_string(),
            ))
        },
    }

    if layouts.is_empty() {
        return Err(HoduError::BackendError(
            "concat requires at least one input".to_string(),
        ));
    }

    if storages.len() != layouts.len() {
        return Err(HoduError::BackendError(
            "storages and layouts length mismatch".to_string(),
        ));
    }

    let first_shape = layouts[0].shape();
    let ndim = first_shape.ndim();

    // Validate dim
    if dim >= ndim {
        return Err(HoduError::InvalidAxis { axis: dim as i32, ndim });
    }

    // Validate all inputs have same dtype and compatible shapes
    let dtype = storages[0].dtype();
    for (i, storage) in storages.iter().enumerate() {
        if storage.dtype() != dtype {
            return Err(HoduError::DTypeMismatch {
                expected: dtype,
                got: storage.dtype(),
            });
        }

        let shape = layouts[i].shape();
        if shape.ndim() != ndim {
            return Err(HoduError::BackendError(format!(
                "all inputs must have same number of dimensions, expected {}, got {}",
                ndim,
                shape.ndim()
            )));
        }

        // Check all dimensions except concat_dim match
        for d in 0..ndim {
            if d != dim && shape.dims()[d] != first_shape.dims()[d] {
                return Err(HoduError::BackendError(format!(
                    "dimension {} mismatch: expected {}, got {}",
                    d,
                    first_shape.dims()[d],
                    shape.dims()[d]
                )));
            }
        }
    }

    // Compute output shape
    let mut output_shape_vec = first_shape.dims().to_vec();
    output_shape_vec[dim] = layouts.iter().map(|l| l.shape().dims()[dim]).sum();
    let output_shape = Shape::new(&output_shape_vec);

    let metadata = crate::op_metadatas::concat_metadata(layouts, dim, &output_shape_vec);
    let num_els = output_shape.size();

    let device = first.backend_device();

    // Strategy (matching CPU backend exactly):
    // 1. Pack all input raw data into a temporary buffer (no need to make contiguous!)
    // 2. Pass original layouts (shapes, strides, offsets) to kernel
    // 3. Kernel handles non-contiguous access using strides and offsets
    // 4. Only buffer_offsets indicate position in packed buffer

    // Calculate total size needed: sum of all storage buffer sizes
    // We need to copy the entire raw buffer data (not just shape size, because of offset/strides)
    let element_size = dtype.get_size_in_bytes();
    let total_bytes: usize = storages.iter().map(|s| s.buffer().length()).sum();
    let temp_input_buffer = device.new_buffer(total_bytes / element_size, dtype, "concat_temp_input")?;

    let command_buffer = device.command_buffer()?;

    // Pack all input raw data into temporary buffer using blit
    let blit = command_buffer.blit_command_encoder();
    blit.set_label("concat_pack_inputs");

    let mut buffer_offset_bytes = 0;
    for storage in storages.iter() {
        let byte_size = storage.buffer().length();

        // Copy entire raw buffer data
        blit.copy_from_buffer(
            storage.buffer(),
            0, // Copy from start of buffer
            &temp_input_buffer,
            buffer_offset_bytes,
            byte_size,
        );

        buffer_offset_bytes += byte_size;
    }

    blit.end_encoding();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els, dtype, "concat_output")?;

    // Call concat kernel
    let kernel_name = format!("hodu_metal_concat_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    let input_offset = BufferOffset::zero_offset(&temp_input_buffer);

    kernels::call_ops_concat(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, dtype))
}

pub fn call_ops_split(
    storage: &MetalStorage,
    layout: &Layout,
    dim: usize,
    start: usize,
    size: usize,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    match op {
        Op::Split(_) => (),
        _ => {
            return Err(HoduError::BackendError(
                "Lcall_ops_splitE expects LsplitE op".to_string(),
            ))
        },
    }

    let input_shape = layout.shape();
    let ndim = input_shape.ndim();

    // Validate dim
    if dim >= ndim {
        return Err(HoduError::InvalidAxis { axis: dim as i32, ndim });
    }

    let dim_size = input_shape.dims()[dim];
    if start + size > dim_size {
        return Err(HoduError::BackendError(format!(
            "split range [{}, {}) exceeds dimension size {}",
            start,
            start + size,
            dim_size
        )));
    }

    // Compute output shape
    let mut output_shape_vec = input_shape.dims().to_vec();
    output_shape_vec[dim] = size;
    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();

    let metadata = crate::op_metadatas::split_metadata(layout, dim, size, start, num_els);

    let dtype = storage.dtype();
    let device = storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els, dtype, "split_output")?;

    // Get kernel name
    let kernel_name = format!("hodu_metal_split_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offset for input
    let input_offset = BufferOffset::zero_offset(storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_split(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, dtype))
}
