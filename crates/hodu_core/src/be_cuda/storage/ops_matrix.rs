use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    compat::*,
    error::{HoduError, HoduResult},
    ops::{MatrixOp, Op},
    types::Layout,
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};

pub fn call_ops_matmul(
    lhs_storage: &CudaStorage,
    rhs_storage: &CudaStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<CudaStorage> {
    // Validate op
    match op {
        Op::Matrix(MatrixOp::Matmul) => (),
        _ => return Err(HoduError::BackendError("call_ops_matmul expects Matmul op".to_string())),
    };

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let lhs_ndim = lhs_shape.ndim();
    let rhs_ndim = rhs_shape.ndim();

    let mut output_shape_vec = Vec::new();
    for i in 0..(lhs_ndim - 2) {
        output_shape_vec.push(lhs_shape[i]);
    }
    output_shape_vec.push(lhs_shape[lhs_ndim - 2]);
    output_shape_vec.push(rhs_shape[rhs_ndim - 1]);
    let output_shape = crate::types::Shape::new(&output_shape_vec);
    let output_layout = Layout::from_shape(&output_shape);
    let output_size = output_shape.size();

    let metadata = crate::op_metadatas::matmul_metadata(lhs_layout, rhs_layout, &output_layout)?;

    let dtype = lhs_storage.dtype();
    let device = lhs_storage.get_device();

    let kernel_name = format!("hodu_cuda_matmul_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    macro_rules! call_matmul {
        ($lhs:expr, $rhs:expr, $ty:ty) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(output_size as usize)?;
            kernels::call_ops_matmul(
                kernel,
                device.kernels(),
                device.context(),
                $lhs,
                $rhs,
                &mut output,
                &metadata,
            )?;
            output
        }};
    }

    let device_id = lhs_storage.device_id();
    let device_arc = Arc::clone(&lhs_storage.device);

    match (&lhs_storage.data, &rhs_storage.data) {
        (CudaStorageData::F32(lhs), CudaStorageData::F32(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F32(call_matmul!(lhs, rhs, f32)),
        )),
        #[cfg(feature = "f64")]
        (CudaStorageData::F64(lhs), CudaStorageData::F64(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F64(call_matmul!(lhs, rhs, f64)),
        )),
        (CudaStorageData::F16(lhs), CudaStorageData::F16(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F16(call_matmul!(lhs, rhs, half::f16)),
        )),
        (CudaStorageData::BF16(lhs), CudaStorageData::BF16(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::BF16(call_matmul!(lhs, rhs, half::bf16)),
        )),
        _ => Err(HoduError::DTypeMismatch {
            expected: lhs_storage.dtype(),
            got: rhs_storage.dtype(),
        }),
    }
}

pub fn call_ops_dot(
    lhs_storage: &CudaStorage,
    rhs_storage: &CudaStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<CudaStorage> {
    // Validate op
    match op {
        Op::Matrix(MatrixOp::Dot) => (),
        _ => return Err(HoduError::BackendError("call_ops_dot expects Dot op".to_string())),
    };

    let metadata = crate::op_metadatas::dot_metadata(lhs_layout, rhs_layout)?;

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let m = lhs_shape[0];
    let n = rhs_shape[1];
    let output_size = m * n;

    let dtype = lhs_storage.dtype();
    let device = lhs_storage.get_device();

    let kernel_name = format!("hodu_cuda_dot_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    macro_rules! call_dot {
        ($lhs:expr, $rhs:expr, $ty:ty) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(output_size)?;
            kernels::call_ops_dot(
                kernel,
                device.kernels(),
                device.context(),
                $lhs,
                $rhs,
                &mut output,
                &metadata,
            )?;
            output
        }};
    }

    let device_id = lhs_storage.device_id();
    let device_arc = Arc::clone(&lhs_storage.device);

    match (&lhs_storage.data, &rhs_storage.data) {
        (CudaStorageData::F32(lhs), CudaStorageData::F32(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F32(call_dot!(lhs, rhs, f32)),
        )),
        #[cfg(feature = "f64")]
        (CudaStorageData::F64(lhs), CudaStorageData::F64(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F64(call_dot!(lhs, rhs, f64)),
        )),
        (CudaStorageData::F16(lhs), CudaStorageData::F16(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F16(call_dot!(lhs, rhs, half::f16)),
        )),
        (CudaStorageData::BF16(lhs), CudaStorageData::BF16(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::BF16(call_dot!(lhs, rhs, half::bf16)),
        )),
        _ => Err(HoduError::DTypeMismatch {
            expected: lhs_storage.dtype(),
            got: rhs_storage.dtype(),
        }),
    }
}
