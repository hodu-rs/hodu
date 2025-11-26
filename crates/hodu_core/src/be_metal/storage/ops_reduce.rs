use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    types::Layout,
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_ops_reduce(
    storage: &MetalStorage,
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Extract reduce op
    let reduce_op = match op {
        Op::Reduce(reduce_op) => reduce_op,
        _ => {
            return Err(HoduError::BackendError(
                "Lcall_ops_reduceE expects LreduceE op".to_string(),
            ))
        },
    };

    // Validate reduce dimensions
    for &dim in dims {
        if dim >= layout.shape().ndim() {
            return Err(HoduError::InvalidAxis {
                axis: dim as i32,
                ndim: layout.shape().ndim(),
            });
        }
    }

    let metadata = crate::op_metadatas::reduce_metadata(layout, dims, keep_dim);

    // Compute output size from metadata
    let output_shape_len_idx = 1 + layout.shape().ndim() * 2 + 1;
    let output_shape_len = metadata[output_shape_len_idx];
    let output_shape_start = output_shape_len_idx + 1;
    let output_size: usize = metadata[output_shape_start..output_shape_start + output_shape_len]
        .iter()
        .product();

    let dtype = storage.dtype();
    let device = storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(output_size, dtype, "reduce_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", reduce_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offset for input
    let input_offset = BufferOffset::zero_offset(storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_reduce(
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
