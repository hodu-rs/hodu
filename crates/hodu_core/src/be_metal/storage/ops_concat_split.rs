use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    types::{Layout, Shape},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_concat(
    first: &MetalStorage,
    others: &[&MetalStorage],
    layouts: &[&Layout],
    dim: u32,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Collect all storages
    let mut storages = vec![first];
    storages.extend(others.iter().copied());

    // Validate op
    let _concat_op = match op {
        Op::Concat(_) => (),
        _ => return Err(HoduError::InternalError("call_concat expects concat op".to_string())),
    };

    if layouts.is_empty() {
        return Err(HoduError::InternalError(
            "concat requires at least one input".to_string(),
        ));
    }

    if storages.len() != layouts.len() {
        return Err(HoduError::InternalError(
            "storages and layouts length mismatch".to_string(),
        ));
    }

    let num_inputs = storages.len();
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
            return Err(HoduError::InternalError(format!(
                "all inputs must have same number of dimensions, expected {}, got {}",
                ndim,
                shape.ndim()
            )));
        }

        // Check all dimensions except concat_dim match
        for d in 0..ndim {
            if d != dim && shape.dims()[d as usize] != first_shape.dims()[d as usize] {
                return Err(HoduError::InternalError(format!(
                    "dimension {} mismatch: expected {}, got {}",
                    d,
                    first_shape.dims()[d as usize],
                    shape.dims()[d as usize]
                )));
            }
        }
    }

    // Compute output shape
    let mut output_shape_vec = first_shape.dims().to_vec();
    output_shape_vec[dim as usize] = layouts.iter().map(|l| l.shape().dims()[dim as usize]).sum();
    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();

    // Build metadata for Metal kernel
    let mut metadata = Vec::new();
    metadata.push(num_els as usize);
    metadata.push(ndim as usize);

    // Add output shape
    for &d in &output_shape_vec {
        metadata.push(d as usize);
    }

    metadata.push(dim as usize);
    metadata.push(num_inputs);

    // Add shapes, strides, and offsets for each input
    for i in 0..num_inputs {
        let shape = layouts[i].shape();
        for &d in shape.dims() {
            metadata.push(d as usize);
        }
    }

    for i in 0..num_inputs {
        let strides = layouts[i].strides();
        for &s in strides {
            metadata.push(s as usize);
        }
    }

    for i in 0..num_inputs {
        metadata.push(layouts[i].offset() as usize);
    }

    let device = first.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els as usize, dtype, "concat_output")?;

    // Get kernel name
    let kernel_name = format!("concat_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Add buffer offsets - for now assume all from same underlying allocation
    // TODO: This may need revision if inputs come from different buffers
    for _ in 0..num_inputs {
        metadata.push(0); // buffer offset for each input
    }

    // Use first input's buffer as the combined input buffer
    // Note: This assumes all inputs share the same buffer or are properly offset
    let input_offset = BufferOffset::zero_offset(first.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_concat(
        device.device(),
        &command_buffer,
        device.kernels(),
        kernel,
        input_offset,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(
        output_buffer,
        device.clone(),
        num_els as usize,
        dtype,
    ))
}

pub fn call_split(
    storage: &MetalStorage,
    layout: &Layout,
    dim: u32,
    start: u32,
    size: u32,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    let _split_op = match op {
        Op::Split(_) => (),
        _ => return Err(HoduError::InternalError("call_split expects split op".to_string())),
    };

    let input_shape = layout.shape();
    let ndim = input_shape.ndim();

    // Validate dim
    if dim >= ndim {
        return Err(HoduError::InvalidAxis { axis: dim as i32, ndim });
    }

    let dim_size = input_shape.dims()[dim as usize];
    if start + size > dim_size {
        return Err(HoduError::InternalError(format!(
            "split range [{}, {}) exceeds dimension size {}",
            start,
            start + size,
            dim_size
        )));
    }

    // Compute output shape
    let mut output_shape_vec = input_shape.dims().to_vec();
    output_shape_vec[dim as usize] = size;
    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();

    // Build metadata array for Metal kernel
    let mut metadata = Vec::new();
    metadata.push(num_els as usize);
    metadata.push(ndim as usize);

    // Add input shape
    for &d in input_shape.dims() {
        metadata.push(d as usize);
    }

    // Add input strides
    for &s in layout.strides() {
        metadata.push(s as usize);
    }

    metadata.push(layout.offset() as usize);

    // Add output shape
    for &d in &output_shape_vec {
        metadata.push(d as usize);
    }

    metadata.push(dim as usize);
    metadata.push(start as usize);

    let dtype = storage.dtype();
    let device = storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els as usize, dtype, "split_output")?;

    // Get kernel name
    let kernel_name = format!("split_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offset for input
    let input_offset = BufferOffset::zero_offset(storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_split(
        device.device(),
        &command_buffer,
        device.kernels(),
        kernel,
        input_offset,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(
        output_buffer,
        device.clone(),
        num_els as usize,
        dtype,
    ))
}
