use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::HoduResult,
    op_metadatas,
    types::{DType, Layout},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_topk(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    k: usize,
    last_dim_size: usize,
    outer_size: usize,
    largest: bool,
    sorted: bool,
) -> HoduResult<(MetalStorage, MetalStorage)> {
    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    let output_size = k * outer_size;
    let metadata = op_metadatas::topk_metadata(input_layout, k, last_dim_size, outer_size, largest, sorted);

    let kernel_name = format!("hodu_metal_topk_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    let values_buffer = device.new_buffer(output_size, dtype, "topk_values")?;
    let indices_buffer = device.new_buffer(output_size, DType::I32, "topk_indices")?;

    let input_offset = BufferOffset::zero_offset(input_storage.buffer());
    let command_buffer = device.command_buffer()?;

    kernels::call_topk(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &values_buffer,
        &indices_buffer,
        &metadata,
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let values = MetalStorage::new(values_buffer, device.clone(), output_size, dtype);
    let indices = MetalStorage::new(indices_buffer, device.clone(), output_size, DType::I32);

    Ok((values, indices))
}
