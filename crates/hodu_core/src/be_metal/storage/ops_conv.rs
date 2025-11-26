use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    types::{Layout, Shape},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

#[allow(clippy::too_many_arguments)]
pub fn call_ops_conv(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    weight_storage: &MetalStorage,
    weight_layout: &Layout,
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    let conv_op = match op {
        Op::Conv(conv_op) => conv_op,
        _ => return Err(HoduError::BackendError("call_ops_conv expects conv op".to_string())),
    };

    let input_shape = input_layout.shape();
    let weight_shape = weight_layout.shape();
    let input_ndim = input_shape.ndim();

    // Validate conv dimensions (needs at least [N, C, ...spatial])
    if input_ndim < 3 {
        return Err(HoduError::BackendError(
            "conv requires at least 3D input (N, C, spatial...)".to_string(),
        ));
    }

    let spatial_dims = input_ndim - 2;

    // Compute output spatial dimensions
    let mut output_shape_vec: Vec<usize> = vec![input_shape.dims()[0], weight_shape.dims()[0]];

    for i in 0..spatial_dims {
        let input_size = input_shape.dims()[2 + i];
        let kernel_size = weight_shape.dims()[2 + i];
        let s = stride[i];
        let p = padding[i];
        let d = dilation[i];

        let output_size = (input_size + 2 * p - d * (kernel_size - 1) - 1) / s + 1;
        output_shape_vec.push(output_size);
    }

    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();

    // Generate metadata using centralized function based on spatial dimensions
    let metadata = match spatial_dims {
        1 => crate::op_metadatas::conv1d_metadata(
            input_layout,
            weight_layout,
            stride[0],
            padding[0],
            dilation[0],
            &output_shape_vec,
        ),
        2 => crate::op_metadatas::conv2d_metadata(
            input_layout,
            weight_layout,
            stride,
            padding,
            dilation,
            &output_shape_vec,
        ),
        3 => crate::op_metadatas::conv3d_metadata(
            input_layout,
            weight_layout,
            stride,
            padding,
            dilation,
            &output_shape_vec,
        ),
        _ => {
            return Err(HoduError::BackendError(format!(
                "unsupported spatial dimensions: {}",
                spatial_dims
            )))
        },
    };

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els as usize, dtype, "conv_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", conv_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets
    let input_offset = BufferOffset::zero_offset(input_storage.buffer());
    let weight_offset = BufferOffset::zero_offset(weight_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_conv(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
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

#[allow(clippy::too_many_arguments)]
pub fn call_ops_conv_grad_weight(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    grad_output_storage: &MetalStorage,
    grad_output_layout: &Layout,
    weight_shape: &Shape,
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    let conv_op = match op {
        Op::Conv(conv_op) => conv_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_conv_grad_weight expects conv op".to_string(),
            ))
        },
    };

    let input_shape = input_layout.shape();
    let spatial_dims = input_shape.ndim() - 2;
    let num_els = weight_shape.size();

    let metadata = crate::op_metadatas::conv_grad_weight_metadata(
        input_layout,
        grad_output_layout,
        weight_shape.dims(),
        stride,
        padding,
        dilation,
        spatial_dims,
    );

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els, dtype, "conv_grad_weight_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", conv_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets
    let input_offset = BufferOffset::zero_offset(input_storage.buffer());
    let grad_output_offset = BufferOffset::zero_offset(grad_output_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_conv_grad_weight(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        grad_output_offset,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, dtype))
}
