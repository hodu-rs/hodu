use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    error::HoduResult,
    op_params::{ResizeCoordTransform, ResizeMode, ResizeNearestMode},
    types::Layout,
};
use core::ffi::c_void;

pub fn call_ops_resize(
    storage: &CpuStorage,
    layout: &Layout,
    output_shape: &[usize],
    mode: ResizeMode,
    coord_transform: ResizeCoordTransform,
    nearest_mode: ResizeNearestMode,
) -> HoduResult<CpuStorage> {
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

    let metadata =
        crate::op_metadatas::resize_metadata(layout, output_shape, mode_val, coord_transform_val, nearest_mode_val);

    let output_size: usize = output_shape.iter().product();
    let dtype = storage.dtype();
    let kernel_name = format!("hodu_cpu_resize_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    let mut output = CpuDevice::allocate(output_size, dtype)?;

    macro_rules! call_resize {
        ($input_data:expr, $out_data:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_resize(kernel, input_ptr, out_ptr, &metadata)?;
        }};
    }

    match (storage, &mut output) {
        (CpuStorage::F8E4M3(input), CpuStorage::F8E4M3(out)) => call_resize!(input, out),
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(input), CpuStorage::F8E5M2(out)) => call_resize!(input, out),
        (CpuStorage::BF16(input), CpuStorage::BF16(out)) => call_resize!(input, out),
        (CpuStorage::F16(input), CpuStorage::F16(out)) => call_resize!(input, out),
        (CpuStorage::F32(input), CpuStorage::F32(out)) => call_resize!(input, out),
        #[cfg(feature = "f64")]
        (CpuStorage::F64(input), CpuStorage::F64(out)) => call_resize!(input, out),
        _ => {
            return Err(crate::error::HoduError::BackendError(
                "resize only supports float types".to_string(),
            ))
        },
    }

    Ok(output)
}
