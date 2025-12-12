use crate::{
    be::storage::BackendStorageT, be_metal::storage::MetalStorage, einsum::ParsedEinsum, error::HoduResult,
    types::Layout,
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_ops_einsum(
    storage: &MetalStorage,
    inputs: &[&MetalStorage],
    input_layouts: &[&Layout],
    parsed: &ParsedEinsum,
) -> HoduResult<MetalStorage> {
    let dtype = storage.dtype();
    let device = storage.backend_device();

    let output_shape = parsed.compute_output_shape();
    let num_els = output_shape.size();
    let output_layout = Layout::from_shape(&output_shape);

    let output_buffer = device.new_buffer(num_els, dtype, "einsum_output")?;

    let input_layout_refs: Vec<&Layout> = input_layouts.to_vec();
    let metadata = crate::op_metadatas::einsum_metadata(parsed, &input_layout_refs, &output_layout);

    let kernel_name = format!("hodu_metal_einsum_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Collect all input buffer offsets
    let input_offsets: Vec<BufferOffset> = std::iter::once(storage)
        .chain(inputs.iter().cloned())
        .map(|s| BufferOffset::zero_offset(s.buffer()))
        .collect();

    let command_buffer = device.command_buffer()?;

    kernels::call_ops_einsum(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        &input_offsets,
        &output_buffer,
        &metadata,
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, dtype))
}
