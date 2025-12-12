use crate::{
    be_cuda::storage::{CudaStorage, CudaStorageData},
    error::HoduResult,
    op_params::{ResizeCoordTransform, ResizeMode, ResizeNearestMode},
    types::Layout,
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};
use std::sync::Arc;

pub fn call_ops_resize(
    input_storage: &CudaStorage,
    input_layout: &Layout,
    output_shape: &[usize],
    mode: ResizeMode,
    coord_transform: ResizeCoordTransform,
    nearest_mode: ResizeNearestMode,
) -> HoduResult<CudaStorage> {
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
    let device = input_storage.get_device();

    let kernel_name = format!("hodu_cuda_resize_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    let device_id = input_storage.device_id;
    let device_arc = Arc::clone(&input_storage.device);

    macro_rules! call_resize {
        ($input:expr, $ty:ty) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(output_size)?;
            kernels::call_ops_resize(
                kernel,
                device.kernels(),
                device.context(),
                $input,
                &mut output,
                &metadata,
            )?;
            output
        }};
    }

    let output_data = match &input_storage.data {
        CudaStorageData::F8E4M3(input) => CudaStorageData::F8E4M3(call_resize!(input, float8::F8E4M3)),
        #[cfg(feature = "f8e5m2")]
        CudaStorageData::F8E5M2(input) => CudaStorageData::F8E5M2(call_resize!(input, float8::F8E5M2)),
        CudaStorageData::BF16(input) => CudaStorageData::BF16(call_resize!(input, half::bf16)),
        CudaStorageData::F16(input) => CudaStorageData::F16(call_resize!(input, half::f16)),
        CudaStorageData::F32(input) => CudaStorageData::F32(call_resize!(input, f32)),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => CudaStorageData::F64(call_resize!(input, f64)),
        _ => {
            return Err(crate::error::HoduError::BackendError(
                "resize only supports float types".to_string(),
            ))
        },
    };

    Ok(CudaStorage {
        data: output_data,
        device: device_arc,
        device_id,
    })
}
