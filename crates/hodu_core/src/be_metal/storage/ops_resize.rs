use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::HoduResult,
    op_params::{ResizeCoordTransform, ResizeMode, ResizeNearestMode},
    types::Layout,
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_ops_resize(
    input_storage: &MetalStorage,
    input_layout: &Layout,
    output_shape: &[usize],
    mode: ResizeMode,
    coord_transform: ResizeCoordTransform,
    nearest_mode: ResizeNearestMode,
) -> HoduResult<MetalStorage> {
    let mode_val = match mode {
        ResizeMode::Nearest => 0,
        ResizeMode::Linear => 1,
        ResizeMode::Cubic => 2,
    };

    let coord_transform_val = match coord_transform {
        ResizeCoordTransform::HalfPixel => 0,
        ResizeCoordTransform::Asymmetric => 1,
        ResizeCoordTransform::AlignCorners => 2,
        ResizeCoordTransform::PytorchHalfPixel => 3,
    };

    let nearest_mode_val = match nearest_mode {
        ResizeNearestMode::Floor => 0,
        ResizeNearestMode::Ceil => 1,
        ResizeNearestMode::RoundPreferFloor => 2,
        ResizeNearestMode::RoundPreferCeil => 3,
    };

    let metadata = crate::op_metadatas::resize_metadata(
        input_layout,
        output_shape,
        mode_val,
        coord_transform_val,
        nearest_mode_val,
    );

    let output_size: usize = output_shape.iter().product();
    let dtype = input_storage.dtype();
    let device = input_storage.backend_device();

    let output_buffer = device.new_buffer(output_size, dtype, "resize_output")?;
    let kernel_name = format!("hodu_metal_resize_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    let input_offset = BufferOffset::zero_offset(input_storage.buffer());
    let command_buffer = device.command_buffer()?;

    kernels::call_ops_resize(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        input_offset,
        &output_buffer,
        &metadata,
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(MetalStorage::new(output_buffer, device.clone(), output_size, dtype))
}
