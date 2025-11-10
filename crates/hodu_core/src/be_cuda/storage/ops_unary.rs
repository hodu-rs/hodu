use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    scalar::Scalar,
    types::Layout,
};
use hodu_cuda_kernels::{cuda::CudaSlice, kernels};

pub fn call_ops_unary(input_storage: &CudaStorage, layout: &Layout, op: Op) -> HoduResult<CudaStorage> {
    let unary_op = match op {
        Op::Unary(unary_op) => unary_op,
        _ => return Err(HoduError::BackendError("call_ops_unary expects unary op".to_string())),
    };

    let shape = layout.shape();
    let num_els = shape.size();
    let num_dims = shape.ndim();

    let mut metadata = Vec::with_capacity(2 + 2 * num_dims as usize + 1);
    metadata.push(num_els as usize);
    metadata.push(num_dims as usize);

    for &dim in shape.dims() {
        metadata.push(dim as usize);
    }
    for &stride in layout.strides() {
        metadata.push(stride as usize);
    }
    metadata.push(layout.offset() as usize);

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();

    let kernel_name = format!("{}_{}", unary_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    macro_rules! call_unary {
        ($input:expr, $ty:ty) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(num_els as usize)?;
            kernels::call_ops_unary(
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
        CudaStorageData::BOOL(input) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::BOOL(call_unary!(input, bool)),
        )),
        CudaStorageData::F8E4M3(input) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::F8E4M3(call_unary!(input, float8::F8E4M3)),
        )),
        #[cfg(feature = "f8e5m2")]
        CudaStorageData::F8E5M2(input) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::F8E5M2(call_unary!(input, float8::F8E5M2)),
        )),
        CudaStorageData::BF16(input) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::BF16(call_unary!(input, half::bf16)),
        )),
        CudaStorageData::F16(input) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::F16(call_unary!(input, half::f16)),
        )),
        CudaStorageData::F32(input) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::F32(call_unary!(input, f32)),
        )),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::F64(call_unary!(input, f64)),
        )),
        CudaStorageData::U8(input) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::U8(call_unary!(input, u8)),
        )),
        #[cfg(feature = "u16")]
        CudaStorageData::U16(input) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::U16(call_unary!(input, u16)),
        )),
        CudaStorageData::U32(input) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::U32(call_unary!(input, u32)),
        )),
        #[cfg(feature = "u64")]
        CudaStorageData::U64(input) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::U64(call_unary!(input, u64)),
        )),
        CudaStorageData::I8(input) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::I8(call_unary!(input, i8)),
        )),
        #[cfg(feature = "i16")]
        CudaStorageData::I16(input) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::I16(call_unary!(input, i16)),
        )),
        CudaStorageData::I32(input) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::I32(call_unary!(input, i32)),
        )),
        #[cfg(feature = "i64")]
        CudaStorageData::I64(input) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::I64(call_unary!(input, i64)),
        )),
    }
}

pub fn call_ops_unary_logical(input_storage: &CudaStorage, layout: &Layout, op: Op) -> HoduResult<CudaStorage> {
    let unary_op = match op {
        Op::UnaryLogical(unary_op) => unary_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_unary_logical expects unary logical op".to_string(),
            ))
        },
    };

    let shape = layout.shape();
    let num_els = shape.size();
    let num_dims = shape.ndim();

    let mut metadata = Vec::with_capacity(2 + 2 * num_dims as usize + 1);
    metadata.push(num_els as usize);
    metadata.push(num_dims as usize);

    for &dim in shape.dims() {
        metadata.push(dim as usize);
    }
    for &stride in layout.strides() {
        metadata.push(stride as usize);
    }
    metadata.push(layout.offset() as usize);

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();

    let kernel_name = format!("{}_{}", unary_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    macro_rules! call_unary_logical {
        ($input:expr, $ty:ty) => {{
            let mut output: CudaSlice<bool> = device.new_buffer(num_els as usize)?;
            kernels::call_ops_unary(
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

    let output = match &input_storage.data {
        CudaStorageData::BOOL(input) => call_unary_logical!(input, bool),
        CudaStorageData::F32(input) => call_unary_logical!(input, f32),
        CudaStorageData::I32(input) => call_unary_logical!(input, i32),
        CudaStorageData::U32(input) => call_unary_logical!(input, u32),
        _ => {
            return Err(HoduError::UnsupportedDTypeForOp {
                dtype: input_storage.dtype(),
                op,
            })
        },
    };

    Ok(CudaStorage::new(
        input_storage.device_id(),
        input_storage.device.clone(),
        CudaStorageData::BOOL(output),
    ))
}

pub fn call_ops_unary_scalar(
    input_storage: &CudaStorage,
    layout: &Layout,
    scalar: Scalar,
    op: Op,
) -> HoduResult<CudaStorage> {
    let unary_op = match op {
        Op::UnaryScalar(unary_op) => unary_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_unary_scalar expects unary scalar op".to_string(),
            ))
        },
    };

    let shape = layout.shape();
    let num_els = shape.size();
    let num_dims = shape.ndim();

    let mut metadata = Vec::with_capacity(2 + 2 * num_dims as usize + 1);
    metadata.push(num_els as usize);
    metadata.push(num_dims as usize);

    for &dim in shape.dims() {
        metadata.push(dim as usize);
    }
    for &stride in layout.strides() {
        metadata.push(stride as usize);
    }
    metadata.push(layout.offset() as usize);

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();

    let kernel_name = format!("{}_{}", unary_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    macro_rules! call_unary_scalar {
        ($input:expr, $scalar_val:expr, $ty:ty) => {{
            let mut output: CudaSlice<$ty> = device.new_buffer(num_els as usize)?;
            kernels::call_ops_unary_scalar(
                kernel,
                device.kernels(),
                device.context(),
                $input,
                &mut output,
                &metadata,
                $scalar_val,
            )?;
            output
        }};
    }

    match (&input_storage.data, scalar) {
        (CudaStorageData::F32(input), Scalar::F32(v)) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::F32(call_unary_scalar!(input, v, f32)),
        )),
        #[cfg(feature = "f64")]
        (CudaStorageData::F64(input), Scalar::F64(v)) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::F64(call_unary_scalar!(input, v, f64)),
        )),
        (CudaStorageData::I32(input), Scalar::I32(v)) => Ok(CudaStorage::new(
            input_storage.device_id(),
            input_storage.device.clone(),
            CudaStorageData::I32(call_unary_scalar!(input, v, i32)),
        )),
        _ => Err(HoduError::DTypeMismatch {
            expected: dtype,
            got: scalar.dtype(),
        }),
    }
}

pub fn call_ops_cmp_scalar(
    input_storage: &CudaStorage,
    layout: &Layout,
    scalar: Scalar,
    op: Op,
) -> HoduResult<CudaStorage> {
    let cmp_op = match op {
        Op::CmpScalar(cmp_op) => cmp_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_cmp_scalar expects cmp scalar op".to_string(),
            ))
        },
    };

    let shape = layout.shape();
    let num_els = shape.size();
    let num_dims = shape.ndim();

    let mut metadata = Vec::with_capacity(2 + 2 * num_dims as usize + 1);
    metadata.push(num_els as usize);
    metadata.push(num_dims as usize);

    for &dim in shape.dims() {
        metadata.push(dim as usize);
    }
    for &stride in layout.strides() {
        metadata.push(stride as usize);
    }
    metadata.push(layout.offset() as usize);

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();

    let kernel_name = format!("{}_{}", cmp_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    macro_rules! call_cmp_scalar {
        ($input:expr, $scalar_val:expr, $ty:ty) => {{
            let mut output: CudaSlice<bool> = device.new_buffer(num_els as usize)?;
            kernels::call_ops_unary_scalar(
                kernel,
                device.kernels(),
                device.context(),
                $input,
                &mut output,
                &metadata,
                $scalar_val,
            )?;
            output
        }};
    }

    let output = match (&input_storage.data, scalar) {
        (CudaStorageData::BOOL(input), Scalar::BOOL(v)) => call_cmp_scalar!(input, v, bool),
        (CudaStorageData::F8E4M3(input), Scalar::F8E4M3(v)) => call_cmp_scalar!(input, v, float8::F8E4M3),
        #[cfg(feature = "f8e5m2")]
        (CudaStorageData::F8E5M2(input), Scalar::F8E5M2(v)) => call_cmp_scalar!(input, v, float8::F8E5M2),
        (CudaStorageData::BF16(input), Scalar::BF16(v)) => call_cmp_scalar!(input, v, half::bf16),
        (CudaStorageData::F16(input), Scalar::F16(v)) => call_cmp_scalar!(input, v, half::f16),
        (CudaStorageData::F32(input), Scalar::F32(v)) => call_cmp_scalar!(input, v, f32),
        #[cfg(feature = "f64")]
        (CudaStorageData::F64(input), Scalar::F64(v)) => call_cmp_scalar!(input, v, f64),
        (CudaStorageData::U8(input), Scalar::U8(v)) => call_cmp_scalar!(input, v, u8),
        #[cfg(feature = "u16")]
        (CudaStorageData::U16(input), Scalar::U16(v)) => call_cmp_scalar!(input, v, u16),
        (CudaStorageData::U32(input), Scalar::U32(v)) => call_cmp_scalar!(input, v, u32),
        #[cfg(feature = "u64")]
        (CudaStorageData::U64(input), Scalar::U64(v)) => call_cmp_scalar!(input, v, u64),
        (CudaStorageData::I8(input), Scalar::I8(v)) => call_cmp_scalar!(input, v, i8),
        #[cfg(feature = "i16")]
        (CudaStorageData::I16(input), Scalar::I16(v)) => call_cmp_scalar!(input, v, i16),
        (CudaStorageData::I32(input), Scalar::I32(v)) => call_cmp_scalar!(input, v, i32),
        #[cfg(feature = "i64")]
        (CudaStorageData::I64(input), Scalar::I64(v)) => call_cmp_scalar!(input, v, i64),
        _ => {
            return Err(HoduError::DTypeMismatch {
                expected: dtype,
                got: scalar.dtype(),
            })
        },
    };

    Ok(CudaStorage::new(
        input_storage.device_id(),
        input_storage.device.clone(),
        CudaStorageData::BOOL(output),
    ))
}
