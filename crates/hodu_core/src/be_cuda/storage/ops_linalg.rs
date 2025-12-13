use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    error::{HoduError, HoduResult},
    types::{Layout, Shape},
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};
use std::sync::Arc;

pub fn call_ops_det(storage: &CudaStorage, layout: &Layout) -> HoduResult<CudaStorage> {
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
    let device = storage.get_device();

    let kernel_name = format!("hodu_cuda_det_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    macro_rules! call_det {
        ($input:expr, $ty:ty) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(batch_size)?;
            kernels::call_ops_det(
                kernel,
                device.kernels(),
                device.context(),
                $input,
                &mut output,
                batch_size,
                &metadata,
            )?;
            output
        }};
    }

    let device_id = storage.device_id();
    let device_arc = Arc::clone(&storage.device);

    match &storage.data {
        CudaStorageData::F32(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F32(call_det!(input, f32)),
        )),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F64(call_det!(input, f64)),
        )),
        CudaStorageData::F16(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F16(call_det!(input, half::f16)),
        )),
        CudaStorageData::BF16(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::BF16(call_det!(input, half::bf16)),
        )),
        _ => Err(HoduError::BackendError(format!(
            "det not supported for dtype {}",
            storage.dtype()
        ))),
    }
}
