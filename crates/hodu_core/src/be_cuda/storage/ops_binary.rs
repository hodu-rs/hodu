use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    types::Layout,
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};

pub fn call_ops_binary(
    lhs_storage: &CudaStorage,
    rhs_storage: &CudaStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<CudaStorage> {
    let binary_op = match op {
        Op::Binary(binary_op) => binary_op,
        _ => return Err(HoduError::BackendError("call_ops_binary expects binary op".to_string())),
    };

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let num_els = lhs_shape.size();
    let num_dims = lhs_shape.ndim();

    let mut metadata: Vec<usize> = Vec::with_capacity(2 + 4 * num_dims as usize + 2);
    metadata.push(num_els as usize);
    metadata.push(num_dims as usize);

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

    let kernel_name = format!("{}_{}", binary_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);
    let device_id = lhs_storage.device_id();
    let device_arc = Arc::clone(&lhs_storage.device);

    macro_rules! call_binary {
        ($lhs:expr, $rhs:expr, $ty:ty) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(num_els as usize)?;
            kernels::call_ops_binary(
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
        (CudaStorageData::BOOL(lhs), CudaStorageData::BOOL(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::BOOL(call_binary!(lhs, rhs, bool)),
        )),
        (CudaStorageData::F8E4M3(lhs), CudaStorageData::F8E4M3(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F8E4M3(call_binary!(lhs, rhs, float8::F8E4M3)),
        )),
        #[cfg(feature = "f8e5m2")]
        (CudaStorageData::F8E5M2(lhs), CudaStorageData::F8E5M2(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F8E5M2(call_binary!(lhs, rhs, float8::F8E5M2)),
        )),
        (CudaStorageData::BF16(lhs), CudaStorageData::BF16(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::BF16(call_binary!(lhs, rhs, half::bf16)),
        )),
        (CudaStorageData::F16(lhs), CudaStorageData::F16(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F16(call_binary!(lhs, rhs, half::f16)),
        )),
        (CudaStorageData::F32(lhs), CudaStorageData::F32(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F32(call_binary!(lhs, rhs, f32)),
        )),
        #[cfg(feature = "f64")]
        (CudaStorageData::F64(lhs), CudaStorageData::F64(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::F64(call_binary!(lhs, rhs, f64)),
        )),
        (CudaStorageData::U8(lhs), CudaStorageData::U8(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U8(call_binary!(lhs, rhs, u8)),
        )),
        #[cfg(feature = "u16")]
        (CudaStorageData::U16(lhs), CudaStorageData::U16(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U16(call_binary!(lhs, rhs, u16)),
        )),
        (CudaStorageData::U32(lhs), CudaStorageData::U32(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U32(call_binary!(lhs, rhs, u32)),
        )),
        #[cfg(feature = "u64")]
        (CudaStorageData::U64(lhs), CudaStorageData::U64(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::U64(call_binary!(lhs, rhs, u64)),
        )),
        (CudaStorageData::I8(lhs), CudaStorageData::I8(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I8(call_binary!(lhs, rhs, i8)),
        )),
        #[cfg(feature = "i16")]
        (CudaStorageData::I16(lhs), CudaStorageData::I16(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I16(call_binary!(lhs, rhs, i16)),
        )),
        (CudaStorageData::I32(lhs), CudaStorageData::I32(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I32(call_binary!(lhs, rhs, i32)),
        )),
        #[cfg(feature = "i64")]
        (CudaStorageData::I64(lhs), CudaStorageData::I64(rhs)) => Ok(CudaStorage::new(
            device_id,
            Arc::clone(&device_arc),
            CudaStorageData::I64(call_binary!(lhs, rhs, i64)),
        )),
        _ => Err(HoduError::DTypeMismatch {
            expected: lhs_storage.dtype(),
            got: rhs_storage.dtype(),
        }),
    }
}

pub fn call_ops_binary_logical(
    lhs_storage: &CudaStorage,
    rhs_storage: &CudaStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<CudaStorage> {
    let binary_op = match op {
        Op::BinaryLogical(binary_op) => binary_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_binary_logical expects binary logical op".to_string(),
            ))
        },
    };

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let num_els = lhs_shape.size();
    let num_dims = lhs_shape.ndim();

    let mut metadata: Vec<usize> = Vec::with_capacity(2 + 4 * num_dims as usize + 2);
    metadata.push(num_els as usize);
    metadata.push(num_dims as usize);

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

    let kernel_name = format!("{}_{}", binary_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    macro_rules! call_binary_logical {
        ($lhs:expr, $rhs:expr, $ty:ty) => {{
            let mut output: CudaSlice<bool> = device.new_buffer(num_els as usize)?;
            kernels::call_ops_binary(
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

    let output = match (&lhs_storage.data, &rhs_storage.data) {
        (CudaStorageData::BOOL(lhs), CudaStorageData::BOOL(rhs)) => call_binary_logical!(lhs, rhs, bool),
        (CudaStorageData::F8E4M3(lhs), CudaStorageData::F8E4M3(rhs)) => call_binary_logical!(lhs, rhs, float8::F8E4M3),
        #[cfg(feature = "f8e5m2")]
        (CudaStorageData::F8E5M2(lhs), CudaStorageData::F8E5M2(rhs)) => call_binary_logical!(lhs, rhs, float8::F8E5M2),
        (CudaStorageData::BF16(lhs), CudaStorageData::BF16(rhs)) => call_binary_logical!(lhs, rhs, half::bf16),
        (CudaStorageData::F16(lhs), CudaStorageData::F16(rhs)) => call_binary_logical!(lhs, rhs, half::f16),
        (CudaStorageData::F32(lhs), CudaStorageData::F32(rhs)) => call_binary_logical!(lhs, rhs, f32),
        #[cfg(feature = "f64")]
        (CudaStorageData::F64(lhs), CudaStorageData::F64(rhs)) => call_binary_logical!(lhs, rhs, f64),
        (CudaStorageData::U8(lhs), CudaStorageData::U8(rhs)) => call_binary_logical!(lhs, rhs, u8),
        #[cfg(feature = "u16")]
        (CudaStorageData::U16(lhs), CudaStorageData::U16(rhs)) => call_binary_logical!(lhs, rhs, u16),
        (CudaStorageData::U32(lhs), CudaStorageData::U32(rhs)) => call_binary_logical!(lhs, rhs, u32),
        #[cfg(feature = "u64")]
        (CudaStorageData::U64(lhs), CudaStorageData::U64(rhs)) => call_binary_logical!(lhs, rhs, u64),
        (CudaStorageData::I8(lhs), CudaStorageData::I8(rhs)) => call_binary_logical!(lhs, rhs, i8),
        #[cfg(feature = "i16")]
        (CudaStorageData::I16(lhs), CudaStorageData::I16(rhs)) => call_binary_logical!(lhs, rhs, i16),
        (CudaStorageData::I32(lhs), CudaStorageData::I32(rhs)) => call_binary_logical!(lhs, rhs, i32),
        #[cfg(feature = "i64")]
        (CudaStorageData::I64(lhs), CudaStorageData::I64(rhs)) => call_binary_logical!(lhs, rhs, i64),
        _ => {
            return Err(HoduError::DTypeMismatch {
                expected: lhs_storage.dtype(),
                got: rhs_storage.dtype(),
            })
        },
    };

    Ok(CudaStorage::new(device_id, device_arc, CudaStorageData::BOOL(output)))
}

pub fn call_ops_cmp(
    lhs_storage: &CudaStorage,
    rhs_storage: &CudaStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<CudaStorage> {
    let cmp_op = match op {
        Op::Cmp(cmp_op) => cmp_op,
        _ => return Err(HoduError::BackendError("call_ops_cmp expects cmp op".to_string())),
    };

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let num_els = lhs_shape.size();
    let num_dims = lhs_shape.ndim();

    let mut metadata: Vec<usize> = Vec::with_capacity(2 + 4 * num_dims as usize + 2);
    metadata.push(num_els as usize);
    metadata.push(num_dims as usize);

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

    let kernel_name = format!("{}_{}", cmp_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    macro_rules! call_cmp {
        ($lhs:expr, $rhs:expr, $ty:ty) => {{
            let mut output: CudaSlice<bool> = device.new_buffer(num_els as usize)?;
            kernels::call_ops_binary(
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

    let output = match (&lhs_storage.data, &rhs_storage.data) {
        (CudaStorageData::BOOL(lhs), CudaStorageData::BOOL(rhs)) => call_cmp!(lhs, rhs, bool),
        (CudaStorageData::F8E4M3(lhs), CudaStorageData::F8E4M3(rhs)) => call_cmp!(lhs, rhs, float8::F8E4M3),
        #[cfg(feature = "f8e5m2")]
        (CudaStorageData::F8E5M2(lhs), CudaStorageData::F8E5M2(rhs)) => call_cmp!(lhs, rhs, float8::F8E5M2),
        (CudaStorageData::BF16(lhs), CudaStorageData::BF16(rhs)) => call_cmp!(lhs, rhs, half::bf16),
        (CudaStorageData::F16(lhs), CudaStorageData::F16(rhs)) => call_cmp!(lhs, rhs, half::f16),
        (CudaStorageData::F32(lhs), CudaStorageData::F32(rhs)) => call_cmp!(lhs, rhs, f32),
        #[cfg(feature = "f64")]
        (CudaStorageData::F64(lhs), CudaStorageData::F64(rhs)) => call_cmp!(lhs, rhs, f64),
        (CudaStorageData::U8(lhs), CudaStorageData::U8(rhs)) => call_cmp!(lhs, rhs, u8),
        #[cfg(feature = "u16")]
        (CudaStorageData::U16(lhs), CudaStorageData::U16(rhs)) => call_cmp!(lhs, rhs, u16),
        (CudaStorageData::U32(lhs), CudaStorageData::U32(rhs)) => call_cmp!(lhs, rhs, u32),
        #[cfg(feature = "u64")]
        (CudaStorageData::U64(lhs), CudaStorageData::U64(rhs)) => call_cmp!(lhs, rhs, u64),
        (CudaStorageData::I8(lhs), CudaStorageData::I8(rhs)) => call_cmp!(lhs, rhs, i8),
        #[cfg(feature = "i16")]
        (CudaStorageData::I16(lhs), CudaStorageData::I16(rhs)) => call_cmp!(lhs, rhs, i16),
        (CudaStorageData::I32(lhs), CudaStorageData::I32(rhs)) => call_cmp!(lhs, rhs, i32),
        #[cfg(feature = "i64")]
        (CudaStorageData::I64(lhs), CudaStorageData::I64(rhs)) => call_cmp!(lhs, rhs, i64),
        _ => {
            return Err(HoduError::DTypeMismatch {
                expected: lhs_storage.dtype(),
                got: rhs_storage.dtype(),
            })
        },
    };

    Ok(CudaStorage::new(device_id, device_arc, CudaStorageData::BOOL(output)))
}
