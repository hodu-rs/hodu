use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    types::Layout,
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};
use smallvec::SmallVec;

pub fn call_ops_reduce(
    input_storage: &CudaStorage,
    layout: &Layout,
    dims: &[u32],
    keep_dim: bool,
    op: Op,
) -> HoduResult<CudaStorage> {
    let reduce_op = match op {
        Op::Reduce(reduce_op) => reduce_op,
        _ => return Err(HoduError::BackendError("call_ops_reduce expects reduce op".to_string())),
    };

    let input_shape = layout.shape();
    let input_ndim = input_shape.ndim();

    let mut output_shape_vec = SmallVec::<[usize; 24]>::new();
    for i in 0..input_ndim {
        if dims.contains(&i) {
            if keep_dim {
                output_shape_vec.push(1);
            }
        } else {
            output_shape_vec.push(input_shape[i] as usize);
        }
    }
    let output_size: u32 = output_shape_vec.iter().map(|&x| x as u32).product();

    let mut metadata = SmallVec::<[usize; 24]>::new();
    metadata.push(input_ndim as usize);

    for &dim in input_shape.dims() {
        metadata.push(dim as usize);
    }
    for &stride in layout.strides() {
        metadata.push(stride as usize);
    }
    metadata.push(layout.offset() as usize);

    for &dim in &output_shape_vec {
        metadata.push(dim as usize);
    }

    metadata.push(dims.len());
    for &dim in dims {
        metadata.push(dim as usize);
    }

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();

    let kernel_name = format!("{}_{}", reduce_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    macro_rules! call_reduce {
        ($input:expr, $ty:ty) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(output_size as usize)?;
            kernels::call_ops_reduce(
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
            CudaStorageData::F32(call_reduce!(input, f32)),
        )),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => Ok(CudaStorage::new(
            input_storage.device_id,
            input_storage.device.clone(),
            CudaStorageData::F64(call_reduce!(input, f64)),
        )),
        CudaStorageData::F16(input) => Ok(CudaStorage::new(
            input_storage.device_id,
            input_storage.device.clone(),
            CudaStorageData::F16(call_reduce!(input, half::f16)),
        )),
        CudaStorageData::BF16(input) => Ok(CudaStorage::new(
            input_storage.device_id,
            input_storage.device.clone(),
            CudaStorageData::BF16(call_reduce!(input, half::bf16)),
        )),
        CudaStorageData::I32(input) => Ok(CudaStorage::new(
            input_storage.device_id,
            input_storage.device.clone(),
            CudaStorageData::I32(call_reduce!(input, i32)),
        )),
        CudaStorageData::U32(input) => Ok(CudaStorage::new(
            input_storage.device_id,
            input_storage.device.clone(),
            CudaStorageData::U32(call_reduce!(input, u32)),
        )),
        _ => Err(HoduError::UnsupportedDTypeForOp {
            dtype: input_storage.dtype(),
            op,
        }),
    }
}
