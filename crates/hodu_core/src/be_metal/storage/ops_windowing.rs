use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    types::{Layout, Shape},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_ops_reduce_window(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    window_shape: &[usize],
    strides: &[usize],
    padding: &[usize],
    op: Op,
) -> HoduResult<MetalStorage> {
    // Extract windowing op
    let windowing_op = match op {
        Op::Windowing(windowing_op) => windowing_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_reduce_window expects windowing op".to_string(),
            ))
        },
    };

    let input_shape = input_layout.shape();
    let input_ndim = input_shape.ndim();

    // Validate that window_shape, strides, and padding have correct length
    let spatial_dims = input_ndim - 2; // Assuming [N, C, ...spatial]
    if window_shape.len() != spatial_dims || strides.len() != spatial_dims || padding.len() != spatial_dims {
        return Err(HoduError::BackendError(
            "window_shape, strides, and padding must match spatial dimensions".to_string(),
        ));
    }

    // Compute output shape
    let mut output_shape_vec: Vec<usize> = vec![input_shape.dims()[0], input_shape.dims()[1]];

    for i in 0..spatial_dims {
        let input_size = input_shape.dims()[2 + i];
        let window_size = window_shape[i];
        let stride = strides[i];
        let pad = padding[i];

        let output_size = (input_size + 2 * pad - window_size) / stride + 1;
        output_shape_vec.push(output_size);
    }

    let output_shape = Shape::new(&output_shape_vec);
    let output_size = output_shape.size();

    // Build metadata: [output_size, num_dims, input_shape..., input_strides..., input_offset, window_shape..., strides..., padding..., output_shape...]
    let mut metadata = Vec::new();
    metadata.push(output_size as usize);
    metadata.push(input_ndim);

    // Add input shape
    for &d in input_shape.dims() {
        metadata.push(d);
    }

    // Add input strides
    for &s in input_layout.strides() {
        metadata.push(s);
    }

    // Add input offset
    metadata.push(input_layout.offset());

    // Add window shape
    for &w in window_shape {
        metadata.push(w);
    }

    // Add strides
    for &s in strides {
        metadata.push(s);
    }

    // Add padding (need to expand to before/after pairs)
    for &p in padding {
        metadata.push(p); // pad_before
        metadata.push(p); // pad_after
    }

    // Add output shape
    for &d in &output_shape_vec {
        metadata.push(d);
    }

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(output_size as usize, dtype, "reduce_window_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", windowing_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offset for input
    let input_offset = BufferOffset::zero_offset(input_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_reduce_window(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(
        output_buffer,
        device.clone(),
        output_size as usize,
        dtype,
    ))
}
