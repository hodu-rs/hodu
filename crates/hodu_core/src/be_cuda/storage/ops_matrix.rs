use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{MatrixOp, Op},
    types::Layout,
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};
use smallvec::SmallVec;

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

    let output_size = {
        let mut dims = SmallVec::<[usize; 24]>::new();
        for i in 0..(lhs_ndim - 2) {
            dims.push(lhs_shape[i] as usize);
        }
        dims.push(lhs_shape[lhs_ndim - 2] as usize);
        dims.push(rhs_shape[rhs_ndim - 1] as usize);
        dims.iter().copied().product::<u32>()
    };

    let mut metadata = SmallVec::<[usize; 24]>::new();
    metadata.push(output_size as usize);
    metadata.push(lhs_ndim as usize);
    metadata.push(rhs_ndim as usize);

    for &dim in lhs_shape.dims() {
        metadata.push(dim as usize);
    }
    for &dim in rhs_shape.dims() {
        metadata.push(dim as usize);
    }
    for &stride in lhs_layout.strides() {
        metadata.push(stride as usize);
    }
    for &stride in rhs_layout.strides() {
        metadata.push(stride as usize);
    }
    metadata.push(lhs_layout.offset() as usize);
    metadata.push(rhs_layout.offset() as usize);

    let dtype = lhs_storage.dtype();
    let device = lhs_storage.get_device();

    let kernel_name = format!("matmul_{}", dtype);
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

    match (&lhs_storage.data, &rhs_storage.data) {
        (CudaStorageData::F32(lhs), CudaStorageData::F32(rhs)) => Ok(CudaStorage::new(
            lhs_storage.device_id(),
            lhs_storage.device.clone(),
            CudaStorageData::F32(call_matmul!(lhs, rhs, f32)),
        )),
        #[cfg(feature = "f64")]
        (CudaStorageData::F64(lhs), CudaStorageData::F64(rhs)) => Ok(CudaStorage::new(
            lhs_storage.device_id(),
            lhs_storage.device.clone(),
            CudaStorageData::F64(call_matmul!(lhs, rhs, f64)),
        )),
        (CudaStorageData::F16(lhs), CudaStorageData::F16(rhs)) => Ok(CudaStorage::new(
            lhs_storage.device_id(),
            lhs_storage.device.clone(),
            CudaStorageData::F16(call_matmul!(lhs, rhs, half::f16)),
        )),
        (CudaStorageData::BF16(lhs), CudaStorageData::BF16(rhs)) => Ok(CudaStorage::new(
            lhs_storage.device_id(),
            lhs_storage.device.clone(),
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

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();

    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let n = rhs_shape[1];

    let mut metadata = SmallVec::<[usize; 24]>::new();
    metadata.push(m as usize);
    metadata.push(k as usize);
    metadata.push(n as usize);

    for &stride in lhs_layout.strides() {
        metadata.push(stride as usize);
    }
    for &stride in rhs_layout.strides() {
        metadata.push(stride as usize);
    }
    metadata.push(lhs_layout.offset() as usize);
    metadata.push(rhs_layout.offset() as usize);

    let dtype = lhs_storage.dtype();
    let device = lhs_storage.get_device();

    let kernel_name = format!("dot_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    let output_size = (m * n) as usize;

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

    match (&lhs_storage.data, &rhs_storage.data) {
        (CudaStorageData::F32(lhs), CudaStorageData::F32(rhs)) => Ok(CudaStorage::new(
            lhs_storage.device_id(),
            lhs_storage.device.clone(),
            CudaStorageData::F32(call_dot!(lhs, rhs, f32)),
        )),
        #[cfg(feature = "f64")]
        (CudaStorageData::F64(lhs), CudaStorageData::F64(rhs)) => Ok(CudaStorage::new(
            lhs_storage.device_id(),
            lhs_storage.device.clone(),
            CudaStorageData::F64(call_dot!(lhs, rhs, f64)),
        )),
        (CudaStorageData::F16(lhs), CudaStorageData::F16(rhs)) => Ok(CudaStorage::new(
            lhs_storage.device_id(),
            lhs_storage.device.clone(),
            CudaStorageData::F16(call_dot!(lhs, rhs, half::f16)),
        )),
        (CudaStorageData::BF16(lhs), CudaStorageData::BF16(rhs)) => Ok(CudaStorage::new(
            lhs_storage.device_id(),
            lhs_storage.device.clone(),
            CudaStorageData::BF16(call_dot!(lhs, rhs, half::bf16)),
        )),
        _ => Err(HoduError::DTypeMismatch {
            expected: lhs_storage.dtype(),
            got: rhs_storage.dtype(),
        }),
    }
}
