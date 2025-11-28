use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaSlice, CudaStorage, CudaStorageData},
    compat::*,
    error::{HoduError, HoduResult},
    ops::Op,
    types::Layout,
};
use hodu_cuda_kernels::kernels;

pub fn call_ops_reduce(
    input_storage: &CudaStorage,
    layout: &Layout,
    dims: &[usize],
    keep_dim: bool,
    op: Op,
) -> HoduResult<CudaStorage> {
    let reduce_op = match op {
        Op::Reduce(reduce_op) => reduce_op,
        _ => return Err(HoduError::BackendError("call_ops_reduce expects reduce op".to_string())),
    };

    let input_shape = layout.shape();
    let input_ndim = input_shape.ndim();

    // Validate reduce dimensions
    for &dim in dims {
        if dim >= input_ndim {
            return Err(HoduError::InvalidAxis {
                axis: dim as i32,
                ndim: input_ndim,
            });
        }
    }

    // Compute output shape
    let mut output_shape_vec = Vec::new();
    for i in 0..input_ndim {
        if dims.contains(&i) {
            if keep_dim {
                output_shape_vec.push(1);
            }
        } else {
            output_shape_vec.push(input_shape[i]);
        }
    }

    // Handle empty output shape (reduce all dimensions without keep_dim)
    if output_shape_vec.is_empty() {
        output_shape_vec.push(1);
    }

    let output_size: u32 = output_shape_vec.iter().map(|&x| x as u32).product();

    let metadata = crate::op_metadatas::reduce_metadata(layout, dims, keep_dim);

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();
    let kernel_name = format!("hodu_cuda_{}_{}", reduce_op, dtype);
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

    let device_id = input_storage.device_id;
    let device_arc = Arc::clone(&input_storage.device);

    match &input_storage.data {
        CudaStorageData::F32(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F32(call_reduce!(input, f32)),
        )),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F64(call_reduce!(input, f64)),
        )),
        CudaStorageData::F16(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F16(call_reduce!(input, half::f16)),
        )),
        CudaStorageData::BF16(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::BF16(call_reduce!(input, half::bf16)),
        )),
        CudaStorageData::I32(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I32(call_reduce!(input, i32)),
        )),
        CudaStorageData::U32(input) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U32(call_reduce!(input, u32)),
        )),
        _ => Err(HoduError::UnsupportedDTypeForOp {
            dtype: input_storage.dtype(),
            op,
        }),
    }
}
