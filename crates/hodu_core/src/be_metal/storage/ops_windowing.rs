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
    let ndim = input_shape.ndim();

    // Validate window_shape, strides, padding dimensions
    if window_shape.len() != ndim {
        return Err(HoduError::BackendError(format!(
            "window_shape length {} does not match tensor ndim {}",
            window_shape.len(),
            ndim
        )));
    }

    if strides.len() != ndim {
        return Err(HoduError::BackendError(format!(
            "strides length {} does not match tensor ndim {}",
            strides.len(),
            ndim
        )));
    }

    if padding.len() != ndim * 2 {
        return Err(HoduError::BackendError(format!(
            "padding length {} does not match tensor ndim * 2 ({})",
            padding.len(),
            ndim * 2
        )));
    }

    // Compute output shape
    let mut output_shape_vec = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let in_size = input_shape.dims()[i];
        let window_size = window_shape[i];
        let stride = strides[i];
        let pad_before = padding[i * 2];
        let pad_after = padding[i * 2 + 1];

        // Output size formula: floor((in_size + pad_before + pad_after - window_size) / stride) + 1
        let padded_size = in_size + pad_before + pad_after;
        if padded_size < window_size {
            return Err(HoduError::BackendError(format!(
                "padded size {} is less than window size {} in dimension {}",
                padded_size, window_size, i
            )));
        }

        let out_size = (padded_size - window_size) / stride + 1;
        output_shape_vec.push(out_size);
    }

    let output_shape = Shape::new(&output_shape_vec);
    let output_size = output_shape.size();

    // Generate metadata using centralized function
    let metadata =
        crate::op_metadatas::reduce_window_metadata(input_layout, window_shape, strides, padding, &output_shape_vec);

    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(output_size, dtype, "reduce_window_output")?;

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

    Ok(MetalStorage::new(output_buffer, device.clone(), output_size, dtype))
}
