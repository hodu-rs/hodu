use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    types::Layout,
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};
use smallvec::{smallvec, SmallVec};

pub fn call_ops_reduce_window(
    input_storage: &CudaStorage,
    input_layout: &Layout,
    window_shape: &[u32],
    strides: &[u32],
    padding: &[u32],
    op: Op,
) -> HoduResult<CudaStorage> {
    let windowing_op = match op {
        Op::Windowing(windowing_op) => windowing_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_reduce_window expects windowing op".to_string(),
            ))
        },
    };

    let input_shape = input_layout.shape();
    let input_ndim = input_shape.ndim();
    let spatial_dims = input_ndim - 2;

    let mut output_shape_vec: SmallVec<[u32; 24]> = smallvec![input_shape[0], input_shape[1]];
    for i in 0..spatial_dims {
        let input_size = input_shape[2 + i];
        let window_size = window_shape[i as usize];
        let stride = strides[i as usize];
        let pad = padding[i as usize];
        let output_size = (input_size + 2 * pad - window_size) / stride + 1;
        output_shape_vec.push(output_size);
    }

    let output_size: u32 = output_shape_vec.iter().product();

    let mut metadata = SmallVec::<[usize; 24]>::new();
    metadata.push(output_size as usize);
    metadata.push(input_ndim as usize);

    for &d in input_shape.dims() {
        metadata.push(d as usize);
    }
    for &s in input_layout.strides() {
        metadata.push(s as usize);
    }
    metadata.push(input_layout.offset() as usize);

    for &w in window_shape {
        metadata.push(w as usize);
    }
    for &s in strides {
        metadata.push(s as usize);
    }
    for &p in padding {
        metadata.push(p as usize);
        metadata.push(p as usize);
    }
    for &d in &output_shape_vec {
        metadata.push(d as usize);
    }

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();

    let kernel_name = format!("{}_{}", windowing_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    macro_rules! call_reduce_window {
        ($input:expr, $ty:ty) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(output_size as usize)?;
            kernels::call_ops_reduce_window(
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

    match &input_storage.data {
        CudaStorageData::F32(input) => Ok(CudaStorage::new(
            input_storage.device_id,
            input_storage.device.clone(),
            CudaStorageData::F32(call_reduce_window!(input, f32)),
        )),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => Ok(CudaStorage::new(
            input_storage.device_id,
            input_storage.device.clone(),
            CudaStorageData::F64(call_reduce_window!(input, f64)),
        )),
        CudaStorageData::F16(input) => Ok(CudaStorage::new(
            input_storage.device_id,
            input_storage.device.clone(),
            CudaStorageData::F16(call_reduce_window!(input, half::f16)),
        )),
        _ => Err(HoduError::UnsupportedDTypeForOp {
            dtype: input_storage.dtype(),
            op,
        }),
    }
}
