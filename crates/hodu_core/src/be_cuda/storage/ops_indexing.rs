use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    error::{HoduError, HoduResult},
    ops::{IndexingOp, Op},
    types::{DType, Layout, Shape},
};
use hodu_cuda_kernels::kernels;
use std::sync::Arc;

pub fn call_ops_index_select(
    input_storage: &CudaStorage,
    input_layout: &Layout,
    indices_storage: &CudaStorage,
    indices_layout: &Layout,
    dim: usize,
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
    output_shape_vec[dim] = num_indices;
    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();

    let metadata = crate::op_metadatas::index_select_metadata(input_layout, dim, num_indices, num_els);

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();
    let device_id = input_storage.device_id();
    let device_arc = Arc::clone(&input_storage.device);

    // Get kernel name
    let kernel_name = format!("hodu_cuda_index_select_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

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
                device_id,
                Arc::clone(&device_arc),
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
    dim: usize,
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

    let metadata = crate::op_metadatas::index_put_metadata(input_layout, values_layout, dim, num_indices, num_els);

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();
    let device_id = input_storage.device_id();
    let device_arc = Arc::clone(&input_storage.device);

    // Get kernel name
    let kernel_name = format!("hodu_cuda_index_put_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

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
                device_id,
                Arc::clone(&device_arc),
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
    dim: usize,
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

    let indices_shape = indices_layout.shape();
    let output_shape = indices_shape.clone();
    let num_els = output_shape.size();

    let metadata = crate::op_metadatas::gather_metadata(input_layout, indices_layout, dim, num_els);

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();
    let device_id = input_storage.device_id();
    let device_arc = Arc::clone(&input_storage.device);

    // Get kernel name
    let kernel_name = format!("hodu_cuda_gather_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

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
                device_id,
                Arc::clone(&device_arc),
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
    dim: usize,
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
        Op::Indexing(
            IndexingOp::Scatter | IndexingOp::ScatterAdd | IndexingOp::ScatterMax | IndexingOp::ScatterMin,
        ) => (),
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_scatter expects scatter-type op".to_string(),
            ))
        },
    }

    let input_shape = input_layout.shape();
    let num_els = input_shape.size();

    // Validate dtypes match
    if input_storage.dtype() != src_storage.dtype() {
        return Err(HoduError::DTypeMismatch {
            expected: input_storage.dtype(),
            got: src_storage.dtype(),
        });
    }

    let metadata = crate::op_metadatas::scatter_metadata(input_layout, indices_layout, src_layout, dim);

    let dtype = input_storage.dtype();
    let device = input_storage.get_device();
    let device_id = input_storage.device_id();
    let device_arc = Arc::clone(&input_storage.device);

    // Get kernel name
    let kernel_name = format!("hodu_cuda_scatter_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

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
                device_id,
                Arc::clone(&device_arc),
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

pub fn call_ops_onehot(
    indices_storage: &CudaStorage,
    indices_layout: &Layout,
    num_classes: usize,
    axis: usize,
    output_dtype: DType,
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
        Op::Indexing(IndexingOp::Onehot) => (),
        _ => return Err(HoduError::BackendError("call_ops_onehot expects Onehot op".to_string())),
    }

    // Generate metadata and output shape using common function
    let (metadata, output_shape_vec) = crate::op_metadatas::onehot_metadata(indices_layout, num_classes, axis);
    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();

    let device = indices_storage.get_device();
    let device_id = indices_storage.device_id();
    let device_arc = Arc::clone(&indices_storage.device);

    // Get kernel name
    let kernel_name = format!("hodu_cuda_onehot_{}", output_dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Extract indices
    let indices = match &indices_storage.data {
        CudaStorageData::I32(data) => data,
        _ => unreachable!(),
    };

    macro_rules! call_kernel {
        ($ty:ty, $variant:ident) => {{
            let mut output = device.new_buffer::<$ty>(num_els as usize)?;
            kernels::call_ops_onehot(
                kernel,
                device.kernels(),
                device.context(),
                indices,
                &mut output,
                &metadata,
            )?;
            Ok(CudaStorage::new(
                device_id,
                Arc::clone(&device_arc),
                CudaStorageData::$variant(output),
            ))
        }};
    }

    match output_dtype {
        DType::BOOL => call_kernel!(bool, BOOL),
        DType::F8E4M3 => call_kernel!(float8::F8E4M3, F8E4M3),
        #[cfg(feature = "f8e5m2")]
        DType::F8E5M2 => call_kernel!(float8::F8E5M2, F8E5M2),
        DType::BF16 => call_kernel!(half::bf16, BF16),
        DType::F16 => call_kernel!(half::f16, F16),
        DType::F32 => call_kernel!(f32, F32),
        #[cfg(feature = "f64")]
        DType::F64 => call_kernel!(f64, F64),
        DType::U8 => call_kernel!(u8, U8),
        #[cfg(feature = "u16")]
        DType::U16 => call_kernel!(u16, U16),
        DType::U32 => call_kernel!(u32, U32),
        #[cfg(feature = "u64")]
        DType::U64 => call_kernel!(u64, U64),
        DType::I8 => call_kernel!(i8, I8),
        #[cfg(feature = "i16")]
        DType::I16 => call_kernel!(i16, I16),
        DType::I32 => call_kernel!(i32, I32),
        #[cfg(feature = "i64")]
        DType::I64 => call_kernel!(i64, I64),
        #[allow(unreachable_patterns)]
        _ => Err(HoduError::UnsupportedDType(output_dtype)),
    }
}

pub fn call_nonzero(input_storage: &CudaStorage, input_layout: &Layout) -> HoduResult<(CudaStorage, usize)> {
    let dtype = input_storage.dtype();
    let ndim = input_layout.ndim();
    let num_els = input_layout.size();

    let device = input_storage.get_device();
    let device_id = input_storage.device_id();
    let device_arc = Arc::clone(&input_storage.device);

    // Generate metadata using common function
    let metadata = crate::op_metadatas::nonzero_metadata(input_layout);

    // Create count buffer (single u32, initialized to 0)
    let mut count_buffer = device.new_buffer::<u32>(1)?;
    device
        .context()
        .default_stream()
        .memcpy_stod(&[0u32])
        .and_then(|zeroed| {
            device
                .context()
                .default_stream()
                .memcpy_dtod(&zeroed, &mut count_buffer)
        })
        .map_err(|e| HoduError::BackendError(format!("Failed to initialize count buffer: {:?}", e)))?;

    // First pass: count non-zero elements
    let count_kernel_name = format!("hodu_cuda_nonzero_count_{}", dtype);
    let count_kernel_name_static = crate::cache::kernel::get_kernel_name(count_kernel_name);
    let count_kernel = kernels::Kernel(count_kernel_name_static);

    macro_rules! call_count_kernel {
        ($input:expr) => {{
            kernels::call_nonzero_count(
                count_kernel,
                device.kernels(),
                device.context(),
                $input,
                &mut count_buffer,
                &metadata,
            )?;
        }};
    }

    match &input_storage.data {
        CudaStorageData::BOOL(input) => call_count_kernel!(input),
        CudaStorageData::F8E4M3(input) => call_count_kernel!(input),
        #[cfg(feature = "f8e5m2")]
        CudaStorageData::F8E5M2(input) => call_count_kernel!(input),
        CudaStorageData::BF16(input) => call_count_kernel!(input),
        CudaStorageData::F16(input) => call_count_kernel!(input),
        CudaStorageData::F32(input) => call_count_kernel!(input),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => call_count_kernel!(input),
        CudaStorageData::U8(input) => call_count_kernel!(input),
        #[cfg(feature = "u16")]
        CudaStorageData::U16(input) => call_count_kernel!(input),
        CudaStorageData::U32(input) => call_count_kernel!(input),
        #[cfg(feature = "u64")]
        CudaStorageData::U64(input) => call_count_kernel!(input),
        CudaStorageData::I8(input) => call_count_kernel!(input),
        #[cfg(feature = "i16")]
        CudaStorageData::I16(input) => call_count_kernel!(input),
        CudaStorageData::I32(input) => call_count_kernel!(input),
        #[cfg(feature = "i64")]
        CudaStorageData::I64(input) => call_count_kernel!(input),
    }

    // Synchronize and read count
    device
        .context()
        .default_stream()
        .synchronize()
        .map_err(|e| HoduError::BackendError(format!("CUDA synchronize failed: {:?}", e)))?;

    let mut count_host = [0u32];
    device
        .context()
        .default_stream()
        .memcpy_dtoh(&count_buffer, &mut count_host)
        .map_err(|e| HoduError::BackendError(format!("Failed to read count: {:?}", e)))?;
    let count = count_host[0] as usize;

    // Handle empty case
    if count == 0 {
        let output = device.new_buffer::<i32>(0)?;
        return Ok((CudaStorage::new(device_id, device_arc, CudaStorageData::I32(output)), 0));
    }

    // Allocate output buffer for [count, ndim] indices
    let output_size = count * ndim;
    let mut output = device.new_buffer::<i32>(output_size)?;

    // Create counter buffer for fill (initialized to 0)
    let mut counter_buffer = device.new_buffer::<u32>(1)?;
    device
        .context()
        .default_stream()
        .memcpy_stod(&[0u32])
        .and_then(|zeroed| {
            device
                .context()
                .default_stream()
                .memcpy_dtod(&zeroed, &mut counter_buffer)
        })
        .map_err(|e| HoduError::BackendError(format!("Failed to initialize counter buffer: {:?}", e)))?;

    // Second pass: fill indices
    let fill_kernel_name = format!("hodu_cuda_nonzero_fill_{}", dtype);
    let fill_kernel_name_static = crate::cache::kernel::get_kernel_name(fill_kernel_name);
    let fill_kernel = kernels::Kernel(fill_kernel_name_static);

    macro_rules! call_fill_kernel {
        ($input:expr) => {{
            kernels::call_nonzero_fill(
                fill_kernel,
                device.kernels(),
                device.context(),
                $input,
                &mut output,
                &mut counter_buffer,
                &metadata,
            )?;
        }};
    }

    match &input_storage.data {
        CudaStorageData::BOOL(input) => call_fill_kernel!(input),
        CudaStorageData::F8E4M3(input) => call_fill_kernel!(input),
        #[cfg(feature = "f8e5m2")]
        CudaStorageData::F8E5M2(input) => call_fill_kernel!(input),
        CudaStorageData::BF16(input) => call_fill_kernel!(input),
        CudaStorageData::F16(input) => call_fill_kernel!(input),
        CudaStorageData::F32(input) => call_fill_kernel!(input),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => call_fill_kernel!(input),
        CudaStorageData::U8(input) => call_fill_kernel!(input),
        #[cfg(feature = "u16")]
        CudaStorageData::U16(input) => call_fill_kernel!(input),
        CudaStorageData::U32(input) => call_fill_kernel!(input),
        #[cfg(feature = "u64")]
        CudaStorageData::U64(input) => call_fill_kernel!(input),
        CudaStorageData::I8(input) => call_fill_kernel!(input),
        #[cfg(feature = "i16")]
        CudaStorageData::I16(input) => call_fill_kernel!(input),
        CudaStorageData::I32(input) => call_fill_kernel!(input),
        #[cfg(feature = "i64")]
        CudaStorageData::I64(input) => call_fill_kernel!(input),
    }

    Ok((
        CudaStorage::new(device_id, device_arc, CudaStorageData::I32(output)),
        count,
    ))
}

/// Unique operation - uses GPU kernels for sorting and finding unique elements
pub fn call_unique(
    input_storage: &CudaStorage,
    input_layout: &Layout,
) -> HoduResult<(CudaStorage, CudaStorage, CudaStorage, usize)> {
    let dtype = input_storage.dtype();
    let shape = input_layout.shape();
    let num_els = shape.size();
    let device_id = input_storage.device_id();
    let device_arc = input_storage.device_arc().clone();
    let context = input_storage.context();
    let kernels = input_storage.kernels();

    if num_els == 0 {
        let stream = context.default_stream();
        let empty_values: CudaSlice<u8> = stream.alloc_zeros(0).map_err(cuda_err)?;
        let empty_inverse: CudaSlice<i32> = stream.alloc_zeros(0).map_err(cuda_err)?;
        let empty_counts: CudaSlice<i32> = stream.alloc_zeros(0).map_err(cuda_err)?;
        return Ok((
            CudaStorage::new(
                device_id,
                device_arc.clone(),
                CudaStorageData::from_slice_u8(empty_values, dtype)?,
            ),
            CudaStorage::new(device_id, device_arc.clone(), CudaStorageData::I32(empty_inverse)),
            CudaStorage::new(device_id, device_arc, CudaStorageData::I32(empty_counts)),
            0,
        ));
    }

    let stream = context.default_stream();

    // Generate metadata using common function
    let metadata = crate::op_metadatas::unique_sort_metadata(input_layout);
    let padded_size = metadata[2];

    macro_rules! call_unique_kernel {
        ($input:expr, $variant:ident, $sort_kernel:expr, $bitonic_kernel:expr, $count_kernel:expr, $mark_kernel:expr, $build_kernel:expr) => {{
            // Step 1: Copy input to sorted_values and initialize sorted_indices (with padding)
            let mut sorted_values: CudaSlice<_> = stream.alloc_zeros(padded_size).map_err(cuda_err)?;
            let mut sorted_indices: CudaSlice<i32> = stream.alloc_zeros(padded_size).map_err(cuda_err)?;

            kernels::call_unique_sort(
                $sort_kernel,
                kernels,
                context,
                $input,
                &mut sorted_values,
                &mut sorted_indices,
                &metadata,
            )?;
            stream.synchronize().map_err(cuda_err)?;

            // Step 2: Bitonic sort
            let mut k = 2;
            while k <= padded_size {
                let mut j = k / 2;
                while j >= 1 {
                    // Generate bitonic step metadata using common function
                    let bitonic_metadata = crate::op_metadatas::unique_bitonic_step_metadata(input_layout, k, j);
                    kernels::call_unique_bitonic_step(
                        $bitonic_kernel,
                        kernels,
                        context,
                        &mut sorted_values,
                        &mut sorted_indices,
                        &bitonic_metadata,
                    )?;
                    stream.synchronize().map_err(cuda_err)?;
                    j /= 2;
                }
                k *= 2;
            }

            // Step 3: Count unique elements
            let mut count_buffer: CudaSlice<u32> = stream.alloc_zeros(1).map_err(cuda_err)?;
            kernels::call_unique_count(
                $count_kernel,
                kernels,
                context,
                &sorted_values,
                &mut count_buffer,
                &metadata,
            )?;
            stream.synchronize().map_err(cuda_err)?;

            let mut count_host = [0u32; 1];
            stream.memcpy_dtoh(&count_buffer, &mut count_host).map_err(cuda_err)?;
            let unique_count = count_host[0] as usize;

            if unique_count == 0 {
                let empty_values: CudaSlice<u8> = stream.alloc_zeros(0).map_err(cuda_err)?;
                let empty_inverse: CudaSlice<i32> = stream.alloc_zeros(0).map_err(cuda_err)?;
                let empty_counts: CudaSlice<i32> = stream.alloc_zeros(0).map_err(cuda_err)?;
                return Ok((
                    CudaStorage::new(
                        device_id,
                        device_arc.clone(),
                        CudaStorageData::from_slice_u8(empty_values, dtype)?,
                    ),
                    CudaStorage::new(device_id, device_arc.clone(), CudaStorageData::I32(empty_inverse)),
                    CudaStorage::new(device_id, device_arc, CudaStorageData::I32(empty_counts)),
                    0,
                ));
            }

            // Step 4: Mark unique boundaries
            let mut marks: CudaSlice<u32> = stream.alloc_zeros(num_els).map_err(cuda_err)?;
            kernels::call_unique_mark($mark_kernel, kernels, context, &sorted_values, &mut marks, &metadata)?;
            stream.synchronize().map_err(cuda_err)?;

            // Step 5: Prefix sum to get unique indices
            let mut unique_idx: CudaSlice<i32> = stream.alloc_zeros(num_els).map_err(cuda_err)?;
            kernels::call_unique_prefix_sum(kernels, context, &marks, &mut unique_idx, &metadata)?;
            stream.synchronize().map_err(cuda_err)?;

            // Step 6: Build output arrays
            let mut values_buffer: CudaSlice<_> = stream.alloc_zeros(unique_count).map_err(cuda_err)?;
            let mut inverse_buffer: CudaSlice<i32> = stream.alloc_zeros(num_els).map_err(cuda_err)?;
            let mut counts_buffer: CudaSlice<i32> = stream.alloc_zeros(unique_count).map_err(cuda_err)?;

            kernels::call_unique_build(
                $build_kernel,
                kernels,
                context,
                &sorted_values,
                &sorted_indices,
                &marks,
                &unique_idx,
                &mut values_buffer,
                &mut inverse_buffer,
                &mut counts_buffer,
                &metadata,
            )?;
            stream.synchronize().map_err(cuda_err)?;

            (
                CudaStorage::new(device_id, device_arc.clone(), CudaStorageData::$variant(values_buffer)),
                CudaStorage::new(device_id, device_arc.clone(), CudaStorageData::I32(inverse_buffer)),
                CudaStorage::new(device_id, device_arc, CudaStorageData::I32(counts_buffer)),
                unique_count,
            )
        }};
    }

    let result = match &input_storage.data {
        CudaStorageData::I8(input) => call_unique_kernel!(
            input,
            I8,
            kernels::unique_sort::I8,
            kernels::unique_bitonic_step::I8,
            kernels::unique_count::I8,
            kernels::unique_mark::I8,
            kernels::unique_build::I8
        ),
        #[cfg(feature = "i16")]
        CudaStorageData::I16(input) => call_unique_kernel!(
            input,
            I16,
            kernels::unique_sort::I16,
            kernels::unique_bitonic_step::I16,
            kernels::unique_count::I16,
            kernels::unique_mark::I16,
            kernels::unique_build::I16
        ),
        CudaStorageData::I32(input) => call_unique_kernel!(
            input,
            I32,
            kernels::unique_sort::I32,
            kernels::unique_bitonic_step::I32,
            kernels::unique_count::I32,
            kernels::unique_mark::I32,
            kernels::unique_build::I32
        ),
        #[cfg(feature = "i64")]
        CudaStorageData::I64(input) => call_unique_kernel!(
            input,
            I64,
            kernels::unique_sort::I64,
            kernels::unique_bitonic_step::I64,
            kernels::unique_count::I64,
            kernels::unique_mark::I64,
            kernels::unique_build::I64
        ),
        CudaStorageData::U8(input) => call_unique_kernel!(
            input,
            U8,
            kernels::unique_sort::U8,
            kernels::unique_bitonic_step::U8,
            kernels::unique_count::U8,
            kernels::unique_mark::U8,
            kernels::unique_build::U8
        ),
        #[cfg(feature = "u16")]
        CudaStorageData::U16(input) => call_unique_kernel!(
            input,
            U16,
            kernels::unique_sort::U16,
            kernels::unique_bitonic_step::U16,
            kernels::unique_count::U16,
            kernels::unique_mark::U16,
            kernels::unique_build::U16
        ),
        CudaStorageData::U32(input) => call_unique_kernel!(
            input,
            U32,
            kernels::unique_sort::U32,
            kernels::unique_bitonic_step::U32,
            kernels::unique_count::U32,
            kernels::unique_mark::U32,
            kernels::unique_build::U32
        ),
        #[cfg(feature = "u64")]
        CudaStorageData::U64(input) => call_unique_kernel!(
            input,
            U64,
            kernels::unique_sort::U64,
            kernels::unique_bitonic_step::U64,
            kernels::unique_count::U64,
            kernels::unique_mark::U64,
            kernels::unique_build::U64
        ),
        CudaStorageData::BF16(input) => call_unique_kernel!(
            input,
            BF16,
            kernels::unique_sort::BF16,
            kernels::unique_bitonic_step::BF16,
            kernels::unique_count::BF16,
            kernels::unique_mark::BF16,
            kernels::unique_build::BF16
        ),
        CudaStorageData::F16(input) => call_unique_kernel!(
            input,
            F16,
            kernels::unique_sort::F16,
            kernels::unique_bitonic_step::F16,
            kernels::unique_count::F16,
            kernels::unique_mark::F16,
            kernels::unique_build::F16
        ),
        CudaStorageData::F32(input) => call_unique_kernel!(
            input,
            F32,
            kernels::unique_sort::F32,
            kernels::unique_bitonic_step::F32,
            kernels::unique_count::F32,
            kernels::unique_mark::F32,
            kernels::unique_build::F32
        ),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => call_unique_kernel!(
            input,
            F64,
            kernels::unique_sort::F64,
            kernels::unique_bitonic_step::F64,
            kernels::unique_count::F64,
            kernels::unique_mark::F64,
            kernels::unique_build::F64
        ),
        _ => {
            return Err(HoduError::UnsupportedDType {
                dtype: dtype.to_string(),
                op: "unique".to_string(),
            })
        },
    };

    Ok(result)
}
