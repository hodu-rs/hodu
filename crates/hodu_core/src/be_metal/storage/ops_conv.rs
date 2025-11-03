use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    types::{Layout, Shape},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_conv(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    weight_storage: &MetalStorage,
    weight_layout: &Layout,
    stride: &[u32],
    padding: &[u32],
    dilation: &[u32],
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    let _conv_op = match op {
        Op::Conv(_) => (),
        _ => return Err(HoduError::InternalError("call_conv expects conv op".to_string())),
    };

    let input_shape = input_layout.shape();
    let weight_shape = weight_layout.shape();
    let input_ndim = input_shape.ndim();

    // Validate conv dimensions (needs at least [N, C, ...spatial])
    if input_ndim < 3 {
        return Err(HoduError::InternalError(
            "conv requires at least 3D input (N, C, spatial...)".to_string(),
        ));
    }

    let spatial_dims = input_ndim - 2;

    // Compute output spatial dimensions
    let mut output_shape_vec = vec![input_shape.dims()[0], weight_shape.dims()[0]];

    for i in 0..spatial_dims {
        let input_size = input_shape.dims()[(2 + i) as usize];
        let kernel_size = weight_shape.dims()[(2 + i) as usize];
        let s = stride[i as usize];
        let p = padding[i as usize];
        let d = dilation[i as usize];

        let output_size = (input_size + 2 * p - d * (kernel_size - 1) - 1) / s + 1;
        output_shape_vec.push(output_size);
    }

    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();

    // Build metadata array
    let mut metadata = Vec::new();
    metadata.push(num_els as usize);
    metadata.push(input_ndim as usize);
    metadata.push(spatial_dims as usize);

    // Add shapes
    for &d in input_shape.dims() {
        metadata.push(d as usize);
    }
    for &d in weight_shape.dims() {
        metadata.push(d as usize);
    }
    for &d in &output_shape_vec {
        metadata.push(d as usize);
    }

    // Add strides
    for &s in input_layout.strides() {
        metadata.push(s as usize);
    }
    for &s in weight_layout.strides() {
        metadata.push(s as usize);
    }

    // Add offsets
    metadata.push(input_layout.offset() as usize);
    metadata.push(weight_layout.offset() as usize);

    // Add conv parameters
    for &s in stride {
        metadata.push(s as usize);
    }
    for &p in padding {
        metadata.push(p as usize);
    }
    for &d in dilation {
        metadata.push(d as usize);
    }

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els as usize, dtype, "conv_output")?;

    // Get kernel name
    let kernel_name = format!("conv_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets
    let input_offset = BufferOffset::zero_offset(input_storage.buffer());
    let weight_offset = BufferOffset::zero_offset(weight_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_conv(
        device.device(),
        &command_buffer,
        device.kernels(),
        kernel,
        input_offset,
        weight_offset,
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

pub fn call_conv_grad_weight(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    grad_output_storage: &MetalStorage,
    grad_output_layout: &Layout,
    weight_shape: &Shape,
    stride: &[u32],
    padding: &[u32],
    dilation: &[u32],
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    let _conv_grad_weight_op = match op {
        Op::Conv(_) => (),
        _ => {
            return Err(HoduError::InternalError(
                "call_conv_grad_weight expects conv op".to_string(),
            ))
        },
    };

    let input_shape = input_layout.shape();
    let grad_output_shape = grad_output_layout.shape();
    let num_els = weight_shape.size();

    // Build metadata array
    let mut metadata = Vec::new();
    metadata.push(num_els as usize);

    let input_ndim = input_shape.ndim();
    let spatial_dims = input_ndim - 2;

    metadata.push(input_ndim as usize);
    metadata.push(spatial_dims as usize);

    // Add shapes
    for &d in input_shape.dims() {
        metadata.push(d as usize);
    }
    for &d in grad_output_shape.dims() {
        metadata.push(d as usize);
    }
    for &d in weight_shape.dims() {
        metadata.push(d as usize);
    }

    // Add strides
    for &s in input_layout.strides() {
        metadata.push(s as usize);
    }
    for &s in grad_output_layout.strides() {
        metadata.push(s as usize);
    }

    // Add offsets
    metadata.push(input_layout.offset() as usize);
    metadata.push(grad_output_layout.offset() as usize);

    // Add conv parameters
    for &s in stride {
        metadata.push(s as usize);
    }
    for &p in padding {
        metadata.push(p as usize);
    }
    for &d in dilation {
        metadata.push(d as usize);
    }

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els as usize, dtype, "conv_grad_weight_output")?;

    // Get kernel name
    let kernel_name = format!("conv_grad_weight_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets
    let input_offset = BufferOffset::zero_offset(input_storage.buffer());
    let grad_output_offset = BufferOffset::zero_offset(grad_output_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_conv_grad_weight(
        device.device(),
        &command_buffer,
        device.kernels(),
        kernel,
        input_offset,
        grad_output_offset,
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
