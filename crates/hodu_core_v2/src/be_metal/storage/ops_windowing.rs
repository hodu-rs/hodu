use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    types::{Layout, Shape},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_reduce_window(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    window_shape: &[u32],
    strides: &[u32],
    padding: &[u32],
    op: Op,
) -> HoduResult<MetalStorage> {
    // Extract windowing op
    let windowing_op = match op {
        Op::Windowing(windowing_op) => windowing_op,
        _ => {
            return Err(HoduError::InternalError(
                "call_reduce_window expects windowing op".to_string(),
            ))
        },
    };

    let input_shape = input_layout.shape();
    let input_ndim = input_shape.ndim();

    // Validate that window_shape, strides, and padding have correct length
    let spatial_dims = input_ndim - 2; // Assuming [N, C, ...spatial]
    if window_shape.len() != spatial_dims as usize
        || strides.len() != spatial_dims as usize
        || padding.len() != spatial_dims as usize
    {
        return Err(HoduError::InternalError(
            "window_shape, strides, and padding must match spatial dimensions".to_string(),
        ));
    }

    // Compute output shape
    let mut output_shape_vec = vec![input_shape.dims()[0], input_shape.dims()[1]];

    for i in 0..spatial_dims {
        let input_size = input_shape.dims()[(2 + i) as usize];
        let window_size = window_shape[i as usize];
        let stride = strides[i as usize];
        let pad = padding[i as usize];

        let output_size = (input_size + 2 * pad - window_size) / stride + 1;
        output_shape_vec.push(output_size);
    }

    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();

    // Build metadata array
    let mut metadata = Vec::new();
    metadata.push(num_els as usize);
    metadata.push(input_ndim as usize);
    metadata.push(spatial_dims as usize);

    // Add input shape
    for &d in input_shape.dims() {
        metadata.push(d as usize);
    }

    // Add output shape
    for &d in &output_shape_vec {
        metadata.push(d as usize);
    }

    // Add input strides and offset
    for &s in input_layout.strides() {
        metadata.push(s as usize);
    }
    metadata.push(input_layout.offset() as usize);

    // Add window parameters
    for &w in window_shape {
        metadata.push(w as usize);
    }
    for &s in strides {
        metadata.push(s as usize);
    }
    for &p in padding {
        metadata.push(p as usize);
    }

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els as usize, dtype, "reduce_window_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", windowing_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offset for input
    let input_offset = BufferOffset::zero_offset(input_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_reduce_window(
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
