use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{IndexingOp, Op},
    types::{DType, Layout, Shape},
};
use hodu_cuda_kernels::kernels;

pub fn call_ops_index_select(
    input_storage: &CudaStorage,
    input_layout: &Layout,
    indices_storage: &CudaStorage,
    indices_layout: &Layout,
    dim: u32,
    op: Op,
) -> HoduResult<CudaStorage> {
    // Validate indices dtype
    if indices_storage.dtype() != DType::I32 {
        return Err(HoduError::DTypeMismatch {
            expected: DType::I32,
            got: indices_storage.dtype(),
        });
    }

    // Validate op
    match op {
        Op::Indexing(IndexingOp::IndexSelect) => (),
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_index_select expects IndexSelect op".to_string(),
            ))
        },
    }

    let input_shape = input_layout.shape();
    let indices_shape = indices_layout.shape();
    let input_ndim = input_shape.ndim();

    // Validate dim
    if dim >= input_ndim {
        return Err(HoduError::InvalidAxis {
            axis: dim as i32,
            ndim: input_ndim,
        });
    }

    // Compute output shape
    let mut output_shape_vec = input_shape.dims().to_vec();
    let num_indices = indices_shape.size();
    output_shape_vec[dim as usize] = num_indices;
    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();

    // Get kernel name
    let kernel_name = format!("index_select_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Build metadata
    let num_dims = input_ndim as usize;
    let mut metadata = Vec::with_capacity(2 + num_dims * 2 + 3);
    metadata.push(num_els as usize);
    metadata.push(num_dims);
    metadata.extend(input_shape.dims().iter().map(|&d| d as usize));
    metadata.extend(input_layout.strides().iter().map(|&s| s as usize));
    metadata.push(input_layout.offset() as usize);
    metadata.push(dim as usize);
    metadata.push(num_indices as usize);

    // Extract indices
    let indices = match &indices_storage.data {
        CudaStorageData::I32(data) => data,
        _ => unreachable!(),
    };

    macro_rules! call_kernel {
        ($input:expr, $ty:ty, $variant:ident) => {{
            let mut output = device.new_buffer::<$ty>(num_els as usize)?;
            kernels::call_ops_index_select(
                kernel,
                device.kernels(),
                device.context(),
                $input,
                indices,
                &mut output,
                &metadata,
            )?;
            Ok(CudaStorage::new(
                input_storage.device_id(),
                device.clone(),
                CudaStorageData::$variant(output),
            ))
        }};
    }

    match &input_storage.data {
        CudaStorageData::BOOL(input) => call_kernel!(input, bool, BOOL),
        CudaStorageData::F8E4M3(input) => call_kernel!(input, float8::F8E4M3, F8E4M3),
        #[cfg(feature = "f8e5m2")]
        CudaStorageData::F8E5M2(input) => call_kernel!(input, float8::F8E5M2, F8E5M2),
        CudaStorageData::BF16(input) => call_kernel!(input, half::bf16, BF16),
        CudaStorageData::F16(input) => call_kernel!(input, half::f16, F16),
        CudaStorageData::F32(input) => call_kernel!(input, f32, F32),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => call_kernel!(input, f64, F64),
        CudaStorageData::U8(input) => call_kernel!(input, u8, U8),
        #[cfg(feature = "u16")]
        CudaStorageData::U16(input) => call_kernel!(input, u16, U16),
        CudaStorageData::U32(input) => call_kernel!(input, u32, U32),
        #[cfg(feature = "u64")]
        CudaStorageData::U64(input) => call_kernel!(input, u64, U64),
        CudaStorageData::I8(input) => call_kernel!(input, i8, I8),
        #[cfg(feature = "i16")]
        CudaStorageData::I16(input) => call_kernel!(input, i16, I16),
        CudaStorageData::I32(input) => call_kernel!(input, i32, I32),
        #[cfg(feature = "i64")]
        CudaStorageData::I64(input) => call_kernel!(input, i64, I64),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_ops_index_put(
    input_storage: &CudaStorage,
    input_layout: &Layout,
    indices_storage: &CudaStorage,
    indices_layout: &Layout,
    values_storage: &CudaStorage,
    values_layout: &Layout,
    dim: u32,
    op: Op,
) -> HoduResult<CudaStorage> {
    // Validate indices dtype
    if indices_storage.dtype() != DType::I32 {
        return Err(HoduError::DTypeMismatch {
            expected: DType::I32,
            got: indices_storage.dtype(),
        });
    }

    // Validate op
    match op {
        Op::Indexing(IndexingOp::IndexPut) => (),
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_index_put expects IndexPut op".to_string(),
            ))
        },
    }

    let input_shape = input_layout.shape();
    let indices_shape = indices_layout.shape();
    let num_els = input_shape.size();
    let num_indices = indices_shape.size();

    // Validate dtypes match
    if input_storage.dtype() != values_storage.dtype() {
        return Err(HoduError::DTypeMismatch {
            expected: input_storage.dtype(),
            got: values_storage.dtype(),
        });
    }

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();

    // Get kernel name
    let kernel_name = format!("index_put_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Build metadata
    let num_dims = input_shape.ndim() as usize;
    let mut metadata = Vec::with_capacity(2 + num_dims * 3 + 4);
    metadata.push(num_els as usize);
    metadata.push(num_dims);
    metadata.extend(input_shape.dims().iter().map(|&d| d as usize));
    metadata.extend(input_layout.strides().iter().map(|&s| s as usize));
    metadata.extend(values_layout.strides().iter().map(|&s| s as usize));
    metadata.push(input_layout.offset() as usize);
    metadata.push(values_layout.offset() as usize);
    metadata.push(dim as usize);
    metadata.push(num_indices as usize);

    // Extract indices
    let indices = match &indices_storage.data {
        CudaStorageData::I32(data) => data,
        _ => unreachable!(),
    };

    macro_rules! call_kernel {
        ($input:expr, $values:expr, $ty:ty, $variant:ident) => {{
            // Copy input to output first
            let stream = device.context().default_stream();
            let mut temp = vec![unsafe { core::mem::zeroed() }; $input.len()];
            stream
                .memcpy_dtoh($input, &mut temp)
                .map_err(|e| HoduError::BackendError(format!("CUDA memcpy_dtoh failed: {:?}", e)))?;
            let input_copy = device.new_buffer_with_data(&temp)?;

            let mut output = device.new_buffer::<$ty>(num_els as usize)?;
            kernels::call_ops_index_put(
                kernel,
                device.kernels(),
                device.context(),
                &input_copy,
                indices,
                $values,
                &mut output,
                &metadata,
            )?;
            Ok(CudaStorage::new(
                input_storage.device_id(),
                device.clone(),
                CudaStorageData::$variant(output),
            ))
        }};
    }

    match (&input_storage.data, &values_storage.data) {
        (CudaStorageData::BOOL(input), CudaStorageData::BOOL(values)) => call_kernel!(input, values, bool, BOOL),
        (CudaStorageData::F8E4M3(input), CudaStorageData::F8E4M3(values)) => {
            call_kernel!(input, values, float8::F8E4M3, F8E4M3)
        },
        #[cfg(feature = "f8e5m2")]
        (CudaStorageData::F8E5M2(input), CudaStorageData::F8E5M2(values)) => {
            call_kernel!(input, values, float8::F8E5M2, F8E5M2)
        },
        (CudaStorageData::BF16(input), CudaStorageData::BF16(values)) => call_kernel!(input, values, half::bf16, BF16),
        (CudaStorageData::F16(input), CudaStorageData::F16(values)) => call_kernel!(input, values, half::f16, F16),
        (CudaStorageData::F32(input), CudaStorageData::F32(values)) => call_kernel!(input, values, f32, F32),
        #[cfg(feature = "f64")]
        (CudaStorageData::F64(input), CudaStorageData::F64(values)) => call_kernel!(input, values, f64, F64),
        (CudaStorageData::U8(input), CudaStorageData::U8(values)) => call_kernel!(input, values, u8, U8),
        #[cfg(feature = "u16")]
        (CudaStorageData::U16(input), CudaStorageData::U16(values)) => call_kernel!(input, values, u16, U16),
        (CudaStorageData::U32(input), CudaStorageData::U32(values)) => call_kernel!(input, values, u32, U32),
        #[cfg(feature = "u64")]
        (CudaStorageData::U64(input), CudaStorageData::U64(values)) => call_kernel!(input, values, u64, U64),
        (CudaStorageData::I8(input), CudaStorageData::I8(values)) => call_kernel!(input, values, i8, I8),
        #[cfg(feature = "i16")]
        (CudaStorageData::I16(input), CudaStorageData::I16(values)) => call_kernel!(input, values, i16, I16),
        (CudaStorageData::I32(input), CudaStorageData::I32(values)) => call_kernel!(input, values, i32, I32),
        #[cfg(feature = "i64")]
        (CudaStorageData::I64(input), CudaStorageData::I64(values)) => call_kernel!(input, values, i64, I64),
        _ => Err(HoduError::BackendError(
            "mismatched storage types in index_put".to_string(),
        )),
    }
}

pub fn call_ops_gather(
    input_storage: &CudaStorage,
    input_layout: &Layout,
    indices_storage: &CudaStorage,
    indices_layout: &Layout,
    dim: u32,
    op: Op,
) -> HoduResult<CudaStorage> {
    // Validate indices dtype
    if indices_storage.dtype() != DType::I32 {
        return Err(HoduError::DTypeMismatch {
            expected: DType::I32,
            got: indices_storage.dtype(),
        });
    }

    // Validate op
    match op {
        Op::Indexing(IndexingOp::Gather) => (),
        _ => return Err(HoduError::BackendError("call_ops_gather expects Gather op".to_string())),
    }

    let input_shape = input_layout.shape();
    let indices_shape = indices_layout.shape();
    let output_shape = indices_shape.clone();
    let num_els = output_shape.size();

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();

    // Get kernel name
    let kernel_name = format!("gather_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Build metadata
    let num_dims = input_shape.ndim() as usize;
    let num_indices = indices_shape.size() as usize;
    let mut metadata = Vec::with_capacity(2 + num_dims * 3 + 4);
    metadata.push(num_els as usize);
    metadata.push(num_dims);
    metadata.extend(input_shape.dims().iter().map(|&d| d as usize));
    metadata.extend(input_layout.strides().iter().map(|&s| s as usize));
    metadata.extend(indices_layout.strides().iter().map(|&s| s as usize));
    metadata.push(input_layout.offset() as usize);
    metadata.push(indices_layout.offset() as usize);
    metadata.push(dim as usize);
    metadata.push(num_indices);

    // Extract indices
    let indices = match &indices_storage.data {
        CudaStorageData::I32(data) => data,
        _ => unreachable!(),
    };

    macro_rules! call_kernel {
        ($input:expr, $ty:ty, $variant:ident) => {{
            let mut output = device.new_buffer::<$ty>(num_els as usize)?;
            kernels::call_ops_gather(
                kernel,
                device.kernels(),
                device.context(),
                $input,
                indices,
                &mut output,
                &metadata,
            )?;
            Ok(CudaStorage::new(
                input_storage.device_id(),
                device.clone(),
                CudaStorageData::$variant(output),
            ))
        }};
    }

    match &input_storage.data {
        CudaStorageData::BOOL(input) => call_kernel!(input, bool, BOOL),
        CudaStorageData::F8E4M3(input) => call_kernel!(input, float8::F8E4M3, F8E4M3),
        #[cfg(feature = "f8e5m2")]
        CudaStorageData::F8E5M2(input) => call_kernel!(input, float8::F8E5M2, F8E5M2),
        CudaStorageData::BF16(input) => call_kernel!(input, half::bf16, BF16),
        CudaStorageData::F16(input) => call_kernel!(input, half::f16, F16),
        CudaStorageData::F32(input) => call_kernel!(input, f32, F32),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => call_kernel!(input, f64, F64),
        CudaStorageData::U8(input) => call_kernel!(input, u8, U8),
        #[cfg(feature = "u16")]
        CudaStorageData::U16(input) => call_kernel!(input, u16, U16),
        CudaStorageData::U32(input) => call_kernel!(input, u32, U32),
        #[cfg(feature = "u64")]
        CudaStorageData::U64(input) => call_kernel!(input, u64, U64),
        CudaStorageData::I8(input) => call_kernel!(input, i8, I8),
        #[cfg(feature = "i16")]
        CudaStorageData::I16(input) => call_kernel!(input, i16, I16),
        CudaStorageData::I32(input) => call_kernel!(input, i32, I32),
        #[cfg(feature = "i64")]
        CudaStorageData::I64(input) => call_kernel!(input, i64, I64),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_ops_scatter(
    input_storage: &CudaStorage,
    input_layout: &Layout,
    indices_storage: &CudaStorage,
    indices_layout: &Layout,
    src_storage: &CudaStorage,
    src_layout: &Layout,
    dim: u32,
    op: Op,
) -> HoduResult<CudaStorage> {
    // Validate indices dtype
    if indices_storage.dtype() != DType::I32 {
        return Err(HoduError::DTypeMismatch {
            expected: DType::I32,
            got: indices_storage.dtype(),
        });
    }

    // Validate op
    match op {
        Op::Indexing(IndexingOp::Scatter) => (),
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_scatter expects Scatter op".to_string(),
            ))
        },
    }

    let input_shape = input_layout.shape();
    let src_shape = src_layout.shape();
    let num_els = input_shape.size();

    // Validate dtypes match
    if input_storage.dtype() != src_storage.dtype() {
        return Err(HoduError::DTypeMismatch {
            expected: input_storage.dtype(),
            got: src_storage.dtype(),
        });
    }

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();

    // Get kernel name
    let kernel_name = format!("scatter_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Build metadata
    let num_dims = input_shape.ndim() as usize;
    let num_src_els = src_shape.size() as usize;
    let mut metadata = Vec::with_capacity(2 + num_dims * 5 + 4);
    metadata.push(num_src_els);
    metadata.push(num_dims);
    metadata.extend(input_shape.dims().iter().map(|&d| d as usize));
    metadata.extend(input_layout.strides().iter().map(|&s| s as usize));
    metadata.extend(src_shape.dims().iter().map(|&d| d as usize));
    metadata.extend(src_layout.strides().iter().map(|&s| s as usize));
    metadata.extend(indices_layout.strides().iter().map(|&s| s as usize));
    metadata.push(input_layout.offset() as usize);
    metadata.push(src_layout.offset() as usize);
    metadata.push(indices_layout.offset() as usize);
    metadata.push(dim as usize);

    // Extract indices
    let indices = match &indices_storage.data {
        CudaStorageData::I32(data) => data,
        _ => unreachable!(),
    };

    macro_rules! call_kernel {
        ($input:expr, $src:expr, $ty:ty, $variant:ident) => {{
            // Copy input to output first
            let stream = device.context().default_stream();
            let mut temp = vec![unsafe { core::mem::zeroed() }; $input.len()];
            stream
                .memcpy_dtoh($input, &mut temp)
                .map_err(|e| HoduError::BackendError(format!("CUDA memcpy_dtoh failed: {:?}", e)))?;
            let input_copy = device.new_buffer_with_data(&temp)?;

            let mut output = device.new_buffer::<$ty>(num_els as usize)?;
            kernels::call_ops_scatter(
                kernel,
                device.kernels(),
                device.context(),
                &input_copy,
                indices,
                $src,
                &mut output,
                &metadata,
            )?;
            Ok(CudaStorage::new(
                input_storage.device_id(),
                device.clone(),
                CudaStorageData::$variant(output),
            ))
        }};
    }

    match (&input_storage.data, &src_storage.data) {
        (CudaStorageData::BOOL(input), CudaStorageData::BOOL(src)) => call_kernel!(input, src, bool, BOOL),
        (CudaStorageData::F8E4M3(input), CudaStorageData::F8E4M3(src)) => {
            call_kernel!(input, src, float8::F8E4M3, F8E4M3)
        },
        #[cfg(feature = "f8e5m2")]
        (CudaStorageData::F8E5M2(input), CudaStorageData::F8E5M2(src)) => {
            call_kernel!(input, src, float8::F8E5M2, F8E5M2)
        },
        (CudaStorageData::BF16(input), CudaStorageData::BF16(src)) => call_kernel!(input, src, half::bf16, BF16),
        (CudaStorageData::F16(input), CudaStorageData::F16(src)) => call_kernel!(input, src, half::f16, F16),
        (CudaStorageData::F32(input), CudaStorageData::F32(src)) => call_kernel!(input, src, f32, F32),
        #[cfg(feature = "f64")]
        (CudaStorageData::F64(input), CudaStorageData::F64(src)) => call_kernel!(input, src, f64, F64),
        (CudaStorageData::U8(input), CudaStorageData::U8(src)) => call_kernel!(input, src, u8, U8),
        #[cfg(feature = "u16")]
        (CudaStorageData::U16(input), CudaStorageData::U16(src)) => call_kernel!(input, src, u16, U16),
        (CudaStorageData::U32(input), CudaStorageData::U32(src)) => call_kernel!(input, src, u32, U32),
        #[cfg(feature = "u64")]
        (CudaStorageData::U64(input), CudaStorageData::U64(src)) => call_kernel!(input, src, u64, U64),
        (CudaStorageData::I8(input), CudaStorageData::I8(src)) => call_kernel!(input, src, i8, I8),
        #[cfg(feature = "i16")]
        (CudaStorageData::I16(input), CudaStorageData::I16(src)) => call_kernel!(input, src, i16, I16),
        (CudaStorageData::I32(input), CudaStorageData::I32(src)) => call_kernel!(input, src, i32, I32),
        #[cfg(feature = "i64")]
        (CudaStorageData::I64(input), CudaStorageData::I64(src)) => call_kernel!(input, src, i64, I64),
        _ => Err(HoduError::BackendError(
            "mismatched storage types in scatter".to_string(),
        )),
    }
}
