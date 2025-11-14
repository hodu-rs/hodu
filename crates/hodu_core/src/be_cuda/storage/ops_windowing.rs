use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    types::Layout,
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};

pub fn call_ops_reduce_window(
    input_storage: &CudaStorage,
    input_layout: &Layout,
    window_shape: &[usize],
    strides: &[usize],
    padding: &[usize],
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

    let mut output_shape_vec: Vec<usize> = vec![input_shape[0], input_shape[1]];
    for i in 0..spatial_dims {
        let input_size = input_shape[2 + i];
        let window_size = window_shape[i];
        let stride = strides[i];
        let pad = padding[i];
        let output_size = (input_size + 2 * pad - window_size) / stride + 1;
        output_shape_vec.push(output_size);
    }

    let output_size: usize = output_shape_vec.iter().product();

    let mut metadata = Vec::new();
    metadata.push(output_size);
    metadata.push(input_ndim);

    for &d in input_shape.dims() {
        metadata.push(d);
    }
    for &s in input_layout.strides() {
        metadata.push(s);
    }
    metadata.push(input_layout.offset());

    for &w in window_shape {
        metadata.push(w);
    }
    for &s in strides {
        metadata.push(s);
    }
    for &p in padding {
        metadata.push(p);
        metadata.push(p);
    }
    for &d in &output_shape_vec {
        metadata.push(d);
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

    let device_id = input_storage.device_id;
    let device_arc = Arc::clone(&input_storage.device);

    match &input_storage.data {
        CudaStorageData::F32(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F32(call_reduce_window!(input, f32)),
        )),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F64(call_reduce_window!(input, f64)),
        )),
        CudaStorageData::F16(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F16(call_reduce_window!(input, half::f16)),
        )),
        _ => Err(HoduError::UnsupportedDTypeForOp {
            dtype: input_storage.dtype(),
            op,
        }),
    }
}
