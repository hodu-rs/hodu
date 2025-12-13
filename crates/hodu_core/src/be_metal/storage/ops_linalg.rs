use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    types::{Layout, Shape},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_ops_det(storage: &MetalStorage, layout: &Layout) -> HoduResult<MetalStorage> {
    let shape = layout.shape();
    let ndim = shape.ndim();

    if ndim < 2 {
        return Err(HoduError::BackendError("det requires at least 2D tensor".to_string()));
    }

    let n = shape.dims()[ndim - 1];
    let m = shape.dims()[ndim - 2];

    if n != m {
        return Err(HoduError::BackendError(format!(
            "det requires square matrix, got {}×{}",
            m, n
        )));
    }

    // Compute output shape (batch dimensions only)
    let output_shape = if ndim == 2 {
        Shape::new(&[1])
    } else {
        Shape::new(&shape.dims()[..ndim - 2])
    };

    let batch_size = output_shape.size();
    let metadata = crate::op_metadatas::det_metadata(layout)?;

    let dtype = storage.dtype();
    let device = storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(batch_size, dtype, "det_output")?;

    // Generate kernel name
    let kernel_name = format!("hodu_metal_det_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offset for input
    let input_offset = BufferOffset::zero_offset(storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_det(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &output_buffer,
        batch_size,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), batch_size, dtype))
}
