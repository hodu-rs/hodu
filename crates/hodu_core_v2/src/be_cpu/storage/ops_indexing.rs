use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{IndexingOp, Op},
    types::{DType, Layout, Shape},
};
use core::ffi::c_void;

/// Execute index_select operation to select elements along a dimension
///
/// # Arguments
/// * `storage` - Input tensor storage
/// * `layout` - Input tensor layout
/// * `indices_storage` - Indices tensor storage (must be integer type)
/// * `indices_layout` - Indices tensor layout
/// * `dim` - Dimension along which to select
/// * `op` - The indexing operation
///
/// # Returns
/// Output storage containing selected elements
pub fn call_index_select(
    storage: &CpuStorage,
    layout: &Layout,
    indices_storage: &CpuStorage,
    indices_layout: &Layout,
    dim: u32,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Validate indices dtype
    if indices_storage.dtype() != DType::I32 {
        return Err(HoduError::DTypeMismatch {
            expected: DType::I32,
            got: indices_storage.dtype(),
        });
    }

    // Extract indices
    let indices = match indices_storage {
        CpuStorage::I32(data) => data.as_slice(),
        _ => unreachable!(),
    };
    // Validate op
    match op {
        Op::Indexing(IndexingOp::IndexSelect) => (),
        _ => {
            return Err(HoduError::InternalError(
                "call_index_select expects IndexSelect op".to_string(),
            ))
        },
    };

    let input_shape = layout.shape();
    let ndim = input_shape.ndim();

    // Validate dim
    if dim >= ndim {
        return Err(HoduError::InvalidAxis { axis: dim as i32, ndim });
    }

    // Compute output shape
    let num_indices = indices_layout.shape().size();
    let mut output_shape_vec = input_shape.dims().to_vec();
    output_shape_vec[dim as usize] = num_indices;
    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();

    // Build metadata array
    // Layout: num_els, num_dims, input_shape, input_strides, input_offset, dim, num_indices
    let mut metadata = Vec::with_capacity(2 + ndim as usize + ndim as usize + 1 + 1 + 1);

    metadata.push(num_els as usize);
    metadata.push(ndim as usize);

    for &d in input_shape.dims() {
        metadata.push(d as usize);
    }

    for &s in layout.strides() {
        metadata.push(s as usize);
    }

    metadata.push(layout.offset() as usize);
    metadata.push(dim as usize);
    metadata.push(num_indices as usize);

    // Generate kernel name
    let dtype = storage.dtype();
    let kernel_name = format!("index_select_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let mut output = CpuDevice::zeros(&output_shape, dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($input_data:expr, $out_data:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let indices_ptr = indices.as_ptr();
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::indexing::call_index_select(kernel, input_ptr, indices_ptr, out_ptr, &metadata)?;
        }};
    }

    match (storage, &mut output) {
        (CpuStorage::BOOL(input), CpuStorage::BOOL(out)) => call_kernel!(input, out),
        (CpuStorage::F8E4M3(input), CpuStorage::F8E4M3(out)) => call_kernel!(input, out),
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(input), CpuStorage::F8E5M2(out)) => call_kernel!(input, out),
        (CpuStorage::BF16(input), CpuStorage::BF16(out)) => call_kernel!(input, out),
        (CpuStorage::F16(input), CpuStorage::F16(out)) => call_kernel!(input, out),
        (CpuStorage::F32(input), CpuStorage::F32(out)) => call_kernel!(input, out),
        #[cfg(feature = "f64")]
        (CpuStorage::F64(input), CpuStorage::F64(out)) => call_kernel!(input, out),
        (CpuStorage::U8(input), CpuStorage::U8(out)) => call_kernel!(input, out),
        #[cfg(feature = "u16")]
        (CpuStorage::U16(input), CpuStorage::U16(out)) => call_kernel!(input, out),
        (CpuStorage::U32(input), CpuStorage::U32(out)) => call_kernel!(input, out),
        #[cfg(feature = "u64")]
        (CpuStorage::U64(input), CpuStorage::U64(out)) => call_kernel!(input, out),
        (CpuStorage::I8(input), CpuStorage::I8(out)) => call_kernel!(input, out),
        #[cfg(feature = "i16")]
        (CpuStorage::I16(input), CpuStorage::I16(out)) => call_kernel!(input, out),
        (CpuStorage::I32(input), CpuStorage::I32(out)) => call_kernel!(input, out),
        #[cfg(feature = "i64")]
        (CpuStorage::I64(input), CpuStorage::I64(out)) => call_kernel!(input, out),
        _ => {
            return Err(HoduError::InternalError(
                "mismatched storage types in call_index_select".to_string(),
            ))
        },
    }

    Ok(output)
}

/// Execute put operation to write values to specific indices
///
/// # Arguments
/// * `storage` - Input tensor storage
/// * `layout` - Input tensor layout
/// * `indices_storage` - Indices tensor storage (must be I32)
/// * `indices_layout` - Indices tensor layout
/// * `values_storage` - Values to write
/// * `values_layout` - Values layout
/// * `dim` - Dimension along which to put
/// * `op` - The indexing operation
///
/// # Returns
/// Output storage with values written at indices
pub fn call_index_put(
    storage: &CpuStorage,
    layout: &Layout,
    indices_storage: &CpuStorage,
    indices_layout: &Layout,
    values_storage: &CpuStorage,
    values_layout: &Layout,
    dim: u32,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Validate indices dtype
    if indices_storage.dtype() != DType::I32 {
        return Err(HoduError::DTypeMismatch {
            expected: DType::I32,
            got: indices_storage.dtype(),
        });
    }

    // Extract indices
    let indices = match indices_storage {
        CpuStorage::I32(data) => data.as_slice(),
        _ => unreachable!(),
    };
    // Validate op
    match op {
        Op::Indexing(IndexingOp::IndexPut) => (),
        _ => {
            return Err(HoduError::InternalError(
                "call_index_put expects IndexPut op".to_string(),
            ))
        },
    };

    let input_shape = layout.shape();
    let ndim = input_shape.ndim();

    // Validate dim
    if dim >= ndim {
        return Err(HoduError::InvalidAxis { axis: dim as i32, ndim });
    }

    // Validate dtypes match
    if storage.dtype() != values_storage.dtype() {
        return Err(HoduError::DTypeMismatch {
            expected: storage.dtype(),
            got: values_storage.dtype(),
        });
    }

    let output_shape = input_shape.clone();
    let num_els = output_shape.size();

    // Build metadata array
    // Layout: num_els, num_dims, input_shape, input_strides, input_offset,
    //         values_shape, values_strides, values_offset, dim, num_indices
    let values_shape = values_layout.shape();
    let values_ndim = values_shape.ndim();

    let mut metadata = Vec::with_capacity(
        2 + ndim as usize + ndim as usize + 1 + values_ndim as usize + values_ndim as usize + 1 + 1 + 1,
    );

    metadata.push(num_els as usize);
    metadata.push(ndim as usize);

    for &d in input_shape.dims() {
        metadata.push(d as usize);
    }

    for &s in layout.strides() {
        metadata.push(s as usize);
    }

    metadata.push(layout.offset() as usize);

    // Add values info
    metadata.push(values_ndim as usize);
    for &d in values_shape.dims() {
        metadata.push(d as usize);
    }

    for &s in values_layout.strides() {
        metadata.push(s as usize);
    }

    metadata.push(values_layout.offset() as usize);

    let num_indices = indices_layout.shape().size();
    metadata.push(dim as usize);
    metadata.push(num_indices as usize);

    // Generate kernel name
    let dtype = storage.dtype();
    let kernel_name = format!("index_put_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage (copy of input)
    let mut output = storage.clone();

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($input_data:expr, $values_data:expr, $out_data:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let indices_ptr = indices.as_ptr();
            let values_ptr = $values_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::indexing::call_index_put(kernel, input_ptr, indices_ptr, values_ptr, out_ptr, &metadata)?;
        }};
    }

    match (storage, values_storage, &mut output) {
        (CpuStorage::BOOL(input), CpuStorage::BOOL(values), CpuStorage::BOOL(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::F8E4M3(input), CpuStorage::F8E4M3(values), CpuStorage::F8E4M3(out)) => {
            call_kernel!(input, values, out)
        },
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(input), CpuStorage::F8E5M2(values), CpuStorage::F8E5M2(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::BF16(input), CpuStorage::BF16(values), CpuStorage::BF16(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::F16(input), CpuStorage::F16(values), CpuStorage::F16(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::F32(input), CpuStorage::F32(values), CpuStorage::F32(out)) => {
            call_kernel!(input, values, out)
        },
        #[cfg(feature = "f64")]
        (CpuStorage::F64(input), CpuStorage::F64(values), CpuStorage::F64(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::U8(input), CpuStorage::U8(values), CpuStorage::U8(out)) => {
            call_kernel!(input, values, out)
        },
        #[cfg(feature = "u16")]
        (CpuStorage::U16(input), CpuStorage::U16(values), CpuStorage::U16(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::U32(input), CpuStorage::U32(values), CpuStorage::U32(out)) => {
            call_kernel!(input, values, out)
        },
        #[cfg(feature = "u64")]
        (CpuStorage::U64(input), CpuStorage::U64(values), CpuStorage::U64(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::I8(input), CpuStorage::I8(values), CpuStorage::I8(out)) => {
            call_kernel!(input, values, out)
        },
        #[cfg(feature = "i16")]
        (CpuStorage::I16(input), CpuStorage::I16(values), CpuStorage::I16(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::I32(input), CpuStorage::I32(values), CpuStorage::I32(out)) => {
            call_kernel!(input, values, out)
        },
        #[cfg(feature = "i64")]
        (CpuStorage::I64(input), CpuStorage::I64(values), CpuStorage::I64(out)) => {
            call_kernel!(input, values, out)
        },
        _ => {
            return Err(HoduError::InternalError(
                "mismatched storage types in call_index_put".to_string(),
            ))
        },
    }

    Ok(output)
}

/// Execute gather operation to gather elements using an indices tensor
///
/// # Arguments
/// * `storage` - Input tensor storage
/// * `layout` - Input tensor layout
/// * `indices_storage` - Indices tensor storage (must be I32)
/// * `indices_layout` - Indices tensor layout
/// * `dim` - Dimension along which to gather
/// * `op` - The indexing operation
///
/// # Returns
/// Output storage containing gathered elements
pub fn call_gather(
    storage: &CpuStorage,
    layout: &Layout,
    indices_storage: &CpuStorage,
    indices_layout: &Layout,
    dim: u32,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Validate indices dtype
    if indices_storage.dtype() != DType::I32 {
        return Err(HoduError::DTypeMismatch {
            expected: DType::I32,
            got: indices_storage.dtype(),
        });
    }

    // Extract indices
    let indices = match indices_storage {
        CpuStorage::I32(data) => data.as_slice(),
        _ => unreachable!(),
    };

    // Validate op
    match op {
        Op::Indexing(IndexingOp::Gather) => (),
        _ => return Err(HoduError::InternalError("call_gather expects Gather op".to_string())),
    };

    let input_shape = layout.shape();
    let ndim = input_shape.ndim();
    let indices_shape = indices_layout.shape();

    // Validate dim
    if dim >= ndim {
        return Err(HoduError::InvalidAxis { axis: dim as i32, ndim });
    }

    // Output shape is same as indices shape
    let output_shape = indices_shape.clone();
    let num_els = output_shape.size();

    // Build metadata array
    let indices_ndim = indices_shape.ndim();
    let mut metadata = Vec::with_capacity(
        2 + ndim as usize + ndim as usize + 1 + indices_ndim as usize + indices_ndim as usize + 1 + 1,
    );

    metadata.push(num_els as usize);
    metadata.push(ndim as usize);

    for &d in input_shape.dims() {
        metadata.push(d as usize);
    }

    for &s in layout.strides() {
        metadata.push(s as usize);
    }

    metadata.push(layout.offset() as usize);

    metadata.push(indices_ndim as usize);
    for &d in indices_shape.dims() {
        metadata.push(d as usize);
    }

    for &s in indices_layout.strides() {
        metadata.push(s as usize);
    }

    metadata.push(indices_layout.offset() as usize);
    metadata.push(dim as usize);

    // Generate kernel name
    let dtype = storage.dtype();
    let kernel_name = format!("gather_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let mut output = CpuDevice::zeros(&output_shape, dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($input_data:expr, $out_data:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let indices_ptr = indices.as_ptr();
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::indexing::call_gather(kernel, input_ptr, indices_ptr, out_ptr, &metadata)?;
        }};
    }

    match (storage, &mut output) {
        (CpuStorage::BOOL(input), CpuStorage::BOOL(out)) => call_kernel!(input, out),
        (CpuStorage::F8E4M3(input), CpuStorage::F8E4M3(out)) => call_kernel!(input, out),
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(input), CpuStorage::F8E5M2(out)) => call_kernel!(input, out),
        (CpuStorage::BF16(input), CpuStorage::BF16(out)) => call_kernel!(input, out),
        (CpuStorage::F16(input), CpuStorage::F16(out)) => call_kernel!(input, out),
        (CpuStorage::F32(input), CpuStorage::F32(out)) => call_kernel!(input, out),
        #[cfg(feature = "f64")]
        (CpuStorage::F64(input), CpuStorage::F64(out)) => call_kernel!(input, out),
        (CpuStorage::U8(input), CpuStorage::U8(out)) => call_kernel!(input, out),
        #[cfg(feature = "u16")]
        (CpuStorage::U16(input), CpuStorage::U16(out)) => call_kernel!(input, out),
        (CpuStorage::U32(input), CpuStorage::U32(out)) => call_kernel!(input, out),
        #[cfg(feature = "u64")]
        (CpuStorage::U64(input), CpuStorage::U64(out)) => call_kernel!(input, out),
        (CpuStorage::I8(input), CpuStorage::I8(out)) => call_kernel!(input, out),
        #[cfg(feature = "i16")]
        (CpuStorage::I16(input), CpuStorage::I16(out)) => call_kernel!(input, out),
        (CpuStorage::I32(input), CpuStorage::I32(out)) => call_kernel!(input, out),
        #[cfg(feature = "i64")]
        (CpuStorage::I64(input), CpuStorage::I64(out)) => call_kernel!(input, out),
        _ => {
            return Err(HoduError::InternalError(
                "mismatched storage types in call_gather".to_string(),
            ))
        },
    }

    Ok(output)
}

/// Execute scatter operation to scatter values to specific positions
///
/// # Arguments
/// * `storage` - Input tensor storage
/// * `layout` - Input tensor layout
/// * `indices_storage` - Indices tensor storage (i32)
/// * `indices_layout` - Indices tensor layout
/// * `values_storage` - Values to scatter
/// * `values_layout` - Values layout
/// * `dim` - Dimension along which to scatter
/// * `op` - The indexing operation
///
/// # Returns
/// Output storage with scattered values
pub fn call_scatter(
    storage: &CpuStorage,
    layout: &Layout,
    indices_storage: &CpuStorage,
    indices_layout: &Layout,
    src_storage: &CpuStorage,
    src_layout: &Layout,
    dim: u32,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Validate indices dtype
    if indices_storage.dtype() != DType::I32 {
        return Err(HoduError::DTypeMismatch {
            expected: DType::I32,
            got: indices_storage.dtype(),
        });
    }

    // Extract indices
    let indices = match indices_storage {
        CpuStorage::I32(data) => data.as_slice(),
        _ => unreachable!(),
    };

    // Validate op
    match op {
        Op::Indexing(IndexingOp::Scatter) => (),
        _ => return Err(HoduError::InternalError("call_scatter expects Scatter op".to_string())),
    };

    let input_shape = layout.shape();
    let ndim = input_shape.ndim();
    let indices_shape = indices_layout.shape();
    let src_shape = src_layout.shape();

    // Validate dim
    if dim >= ndim {
        return Err(HoduError::InvalidAxis { axis: dim as i32, ndim });
    }

    // Validate dtypes match
    if storage.dtype() != src_storage.dtype() {
        return Err(HoduError::DTypeMismatch {
            expected: storage.dtype(),
            got: src_storage.dtype(),
        });
    }

    let output_shape = input_shape.clone();
    let num_els = output_shape.size();

    // Build metadata array
    let indices_ndim = indices_shape.ndim();
    let src_ndim = src_shape.ndim();

    let mut metadata = Vec::with_capacity(
        2 + ndim as usize
            + ndim as usize
            + 1
            + indices_ndim as usize
            + indices_ndim as usize
            + 1
            + src_ndim as usize
            + src_ndim as usize
            + 1
            + 1,
    );

    metadata.push(num_els as usize);
    metadata.push(ndim as usize);

    for &d in input_shape.dims() {
        metadata.push(d as usize);
    }

    for &s in layout.strides() {
        metadata.push(s as usize);
    }

    metadata.push(layout.offset() as usize);

    // Add indices info
    metadata.push(indices_ndim as usize);
    for &d in indices_shape.dims() {
        metadata.push(d as usize);
    }

    for &s in indices_layout.strides() {
        metadata.push(s as usize);
    }

    metadata.push(indices_layout.offset() as usize);

    // Add src info
    metadata.push(src_ndim as usize);
    for &d in src_shape.dims() {
        metadata.push(d as usize);
    }

    for &s in src_layout.strides() {
        metadata.push(s as usize);
    }

    metadata.push(src_layout.offset() as usize);
    metadata.push(dim as usize);

    // Generate kernel name
    let dtype = storage.dtype();
    let kernel_name = format!("scatter_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage (copy of input)
    let mut output = storage.clone();

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($input_data:expr, $values_data:expr, $out_data:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let indices_ptr = indices.as_ptr();
            let values_ptr = $values_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::indexing::call_scatter(kernel, input_ptr, indices_ptr, values_ptr, out_ptr, &metadata)?;
        }};
    }

    match (storage, src_storage, &mut output) {
        (CpuStorage::BOOL(input), CpuStorage::BOOL(values), CpuStorage::BOOL(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::F8E4M3(input), CpuStorage::F8E4M3(values), CpuStorage::F8E4M3(out)) => {
            call_kernel!(input, values, out)
        },
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(input), CpuStorage::F8E5M2(values), CpuStorage::F8E5M2(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::BF16(input), CpuStorage::BF16(values), CpuStorage::BF16(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::F16(input), CpuStorage::F16(values), CpuStorage::F16(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::F32(input), CpuStorage::F32(values), CpuStorage::F32(out)) => {
            call_kernel!(input, values, out)
        },
        #[cfg(feature = "f64")]
        (CpuStorage::F64(input), CpuStorage::F64(values), CpuStorage::F64(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::U8(input), CpuStorage::U8(values), CpuStorage::U8(out)) => {
            call_kernel!(input, values, out)
        },
        #[cfg(feature = "u16")]
        (CpuStorage::U16(input), CpuStorage::U16(values), CpuStorage::U16(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::U32(input), CpuStorage::U32(values), CpuStorage::U32(out)) => {
            call_kernel!(input, values, out)
        },
        #[cfg(feature = "u64")]
        (CpuStorage::U64(input), CpuStorage::U64(values), CpuStorage::U64(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::I8(input), CpuStorage::I8(values), CpuStorage::I8(out)) => {
            call_kernel!(input, values, out)
        },
        #[cfg(feature = "i16")]
        (CpuStorage::I16(input), CpuStorage::I16(values), CpuStorage::I16(out)) => {
            call_kernel!(input, values, out)
        },
        (CpuStorage::I32(input), CpuStorage::I32(values), CpuStorage::I32(out)) => {
            call_kernel!(input, values, out)
        },
        #[cfg(feature = "i64")]
        (CpuStorage::I64(input), CpuStorage::I64(values), CpuStorage::I64(out)) => {
            call_kernel!(input, values, out)
        },
        _ => {
            return Err(HoduError::InternalError(
                "mismatched storage types in call_scatter".to_string(),
            ))
        },
    }

    Ok(output)
}
