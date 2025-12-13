use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    error::{HoduError, HoduResult},
    ops::{IndexingOp, Op},
    types::{DType, Layout, Shape},
};
use std::ffi::c_void;

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
pub fn call_ops_index_select(
    storage: &CpuStorage,
    layout: &Layout,
    indices_storage: &CpuStorage,
    indices_layout: &Layout,
    dim: usize,
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
            return Err(HoduError::BackendError(
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
    output_shape_vec[dim] = num_indices;
    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();

    // Generate metadata using centralized function
    let metadata = crate::op_metadatas::index_select_metadata(layout, dim, num_indices, num_els);

    // Generate kernel name
    let dtype = storage.dtype();
    let kernel_name = format!("hodu_cpu_index_select_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let mut output = CpuDevice::allocate(output_shape.size(), dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($input_data:expr, $out_data:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let indices_ptr = indices.as_ptr();
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_index_select(kernel, input_ptr, indices_ptr, out_ptr, &metadata)?;
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
            return Err(HoduError::BackendError(
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
#[allow(clippy::too_many_arguments)]
pub fn call_ops_index_put(
    storage: &CpuStorage,
    layout: &Layout,
    indices_storage: &CpuStorage,
    indices_layout: &Layout,
    values_storage: &CpuStorage,
    values_layout: &Layout,
    dim: usize,
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
            return Err(HoduError::BackendError(
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

    // Generate metadata using centralized function
    let num_indices = indices_layout.shape().size();
    let metadata = crate::op_metadatas::index_put_metadata(layout, values_layout, dim, num_indices, num_els);

    // Generate kernel name
    let dtype = storage.dtype();
    let kernel_name = format!("hodu_cpu_index_put_{}", dtype);
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

            hodu_cpu_kernels::call_ops_index_put(kernel, input_ptr, indices_ptr, values_ptr, out_ptr, &metadata)?;
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
            return Err(HoduError::BackendError(
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
pub fn call_ops_gather(
    storage: &CpuStorage,
    layout: &Layout,
    indices_storage: &CpuStorage,
    indices_layout: &Layout,
    dim: usize,
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
        _ => return Err(HoduError::BackendError("Lcall_gatherE expects LGatherE op".to_string())),
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

    // Generate metadata using centralized function
    let metadata = crate::op_metadatas::gather_metadata(layout, indices_layout, dim, num_els);

    // Generate kernel name
    let dtype = storage.dtype();
    let kernel_name = format!("hodu_cpu_gather_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let mut output = CpuDevice::allocate(output_shape.size(), dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($input_data:expr, $out_data:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let indices_ptr = indices.as_ptr();
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_gather(kernel, input_ptr, indices_ptr, out_ptr, &metadata)?;
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
            return Err(HoduError::BackendError(
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
#[allow(clippy::too_many_arguments)]
pub fn call_ops_scatter(
    storage: &CpuStorage,
    layout: &Layout,
    indices_storage: &CpuStorage,
    indices_layout: &Layout,
    src_storage: &CpuStorage,
    src_layout: &Layout,
    dim: usize,
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
        Op::Indexing(
            IndexingOp::Scatter | IndexingOp::ScatterAdd | IndexingOp::ScatterMax | IndexingOp::ScatterMin,
        ) => (),
        _ => {
            return Err(HoduError::BackendError(
                "Lcall_scatterE expects scatter-type op".to_string(),
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
    if storage.dtype() != src_storage.dtype() {
        return Err(HoduError::DTypeMismatch {
            expected: storage.dtype(),
            got: src_storage.dtype(),
        });
    }

    // Generate metadata using centralized function
    let metadata = crate::op_metadatas::scatter_metadata(layout, indices_layout, src_layout, dim);

    // Generate kernel name
    let dtype = storage.dtype();
    let kernel_name = format!("hodu_cpu_scatter_{}", dtype);
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

            hodu_cpu_kernels::call_ops_scatter(kernel, input_ptr, indices_ptr, values_ptr, out_ptr, &metadata)?;
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
            return Err(HoduError::BackendError(
                "mismatched storage types in call_scatter".to_string(),
            ))
        },
    }

    Ok(output)
}

/// Execute onehot operation to convert indices to one-hot encoded vectors
///
/// # Arguments
/// * `indices_storage` - Indices tensor storage (must be integer type, will be converted to I32)
/// * `indices_layout` - Indices tensor layout
/// * `num_classes` - Number of classes (depth of one-hot dimension)
/// * `axis` - Dimension for one-hot encoding (normalized to positive)
/// * `output_dtype` - Data type for output tensor
/// * `op` - The indexing operation
///
/// # Returns
/// Output storage containing one-hot encoded vectors
pub fn call_ops_onehot(
    indices_storage: &CpuStorage,
    indices_layout: &Layout,
    num_classes: usize,
    axis: usize,
    output_dtype: DType,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Validate op
    match op {
        Op::Indexing(IndexingOp::Onehot) => (),
        _ => return Err(HoduError::BackendError("call_ops_onehot expects Onehot op".to_string())),
    };

    // Convert indices to i32
    let indices: Vec<i32> = match indices_storage {
        CpuStorage::I8(data) => data.iter().map(|&x| x as i32).collect(),
        #[cfg(feature = "i16")]
        CpuStorage::I16(data) => data.iter().map(|&x| x as i32).collect(),
        CpuStorage::I32(data) => data.to_vec(),
        #[cfg(feature = "i64")]
        CpuStorage::I64(data) => data.iter().map(|&x| x as i32).collect(),
        CpuStorage::U8(data) => data.iter().map(|&x| x as i32).collect(),
        #[cfg(feature = "u16")]
        CpuStorage::U16(data) => data.iter().map(|&x| x as i32).collect(),
        CpuStorage::U32(data) => data.iter().map(|&x| x as i32).collect(),
        #[cfg(feature = "u64")]
        CpuStorage::U64(data) => data.iter().map(|&x| x as i32).collect(),
        _ => {
            return Err(HoduError::BackendError(
                "onehot requires integer type indices".to_string(),
            ))
        },
    };

    let input_shape = indices_layout.shape();
    let num_dims_in = input_shape.ndim();
    let num_input_els = input_shape.size();

    // Compute output shape: insert num_classes at axis position
    let mut output_shape_vec = Vec::with_capacity(num_dims_in + 1);
    for (i, &dim) in input_shape.dims().iter().enumerate() {
        if i == axis {
            output_shape_vec.push(num_classes);
        }
        output_shape_vec.push(dim);
    }
    // If axis == num_dims_in (last position)
    if axis == num_dims_in {
        output_shape_vec.push(num_classes);
    }

    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();
    let num_dims_out = output_shape.ndim();

    // Generate metadata
    // - metadata[0]: num_els (total number of output elements)
    // - metadata[1]: num_input_els (total number of input indices)
    // - metadata[2]: num_classes (depth of one-hot dimension)
    // - metadata[3]: axis (dimension for one-hot encoding)
    // - metadata[4]: num_dims_out (number of output dimensions)
    // - metadata[5..5+num_dims_out]: output_shape
    let mut metadata = Vec::with_capacity(5 + num_dims_out);
    metadata.push(num_els);
    metadata.push(num_input_els);
    metadata.push(num_classes);
    metadata.push(axis);
    metadata.push(num_dims_out);
    metadata.extend_from_slice(output_shape.dims());

    // Generate kernel name
    let kernel_name = format!("hodu_cpu_onehot_{}", output_dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let mut output = CpuDevice::allocate(num_els, output_dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($out_data:expr) => {{
            let indices_ptr = indices.as_ptr();
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_onehot(kernel, indices_ptr, out_ptr, &metadata)?;
        }};
    }

    match &mut output {
        CpuStorage::BOOL(out) => call_kernel!(out),
        CpuStorage::F8E4M3(out) => call_kernel!(out),
        #[cfg(feature = "f8e5m2")]
        CpuStorage::F8E5M2(out) => call_kernel!(out),
        CpuStorage::BF16(out) => call_kernel!(out),
        CpuStorage::F16(out) => call_kernel!(out),
        CpuStorage::F32(out) => call_kernel!(out),
        #[cfg(feature = "f64")]
        CpuStorage::F64(out) => call_kernel!(out),
        CpuStorage::U8(out) => call_kernel!(out),
        #[cfg(feature = "u16")]
        CpuStorage::U16(out) => call_kernel!(out),
        CpuStorage::U32(out) => call_kernel!(out),
        #[cfg(feature = "u64")]
        CpuStorage::U64(out) => call_kernel!(out),
        CpuStorage::I8(out) => call_kernel!(out),
        #[cfg(feature = "i16")]
        CpuStorage::I16(out) => call_kernel!(out),
        CpuStorage::I32(out) => call_kernel!(out),
        #[cfg(feature = "i64")]
        CpuStorage::I64(out) => call_kernel!(out),
    }

    Ok(output)
}

/// Execute nonzero operation to find indices of non-zero elements
///
/// # Arguments
/// * `storage` - Input tensor storage
/// * `layout` - Input tensor layout
///
/// # Returns
/// A tuple of (output storage, count) where output is shape [N, ndim] containing
/// the indices of non-zero elements, and count is the number of non-zero elements
pub fn call_nonzero(storage: &CpuStorage, layout: &Layout) -> HoduResult<(CpuStorage, usize)> {
    let dtype = storage.dtype();
    let shape = layout.shape();
    let ndim = shape.ndim();
    let num_els = shape.size();

    // Build metadata:
    // - metadata[0]: num_els (total number of elements in input)
    // - metadata[1]: num_dims (number of dimensions)
    // - metadata[2..2+num_dims]: input_shape
    // - metadata[2+num_dims..2+2*num_dims]: input_strides
    // - metadata[2+2*num_dims]: input_offset
    let mut metadata = Vec::with_capacity(2 + 2 * ndim + 1);
    metadata.push(num_els);
    metadata.push(ndim);
    metadata.extend_from_slice(shape.dims());
    metadata.extend_from_slice(layout.strides());
    metadata.push(layout.offset());

    // Generate kernel names
    let count_kernel_name = format!("hodu_cpu_nonzero_count_{}", dtype);
    let count_kernel_name_static = crate::cache::kernel::get_kernel_name(count_kernel_name);
    let count_kernel = hodu_cpu_kernels::macros::Kernel(count_kernel_name_static);

    let fill_kernel_name = format!("hodu_cpu_nonzero_fill_{}", dtype);
    let fill_kernel_name_static = crate::cache::kernel::get_kernel_name(fill_kernel_name);
    let fill_kernel = hodu_cpu_kernels::macros::Kernel(fill_kernel_name_static);

    // First pass: count non-zero elements
    let input_ptr = storage.as_ptr() as *const c_void;
    let count = hodu_cpu_kernels::call_nonzero_count(count_kernel, input_ptr, &metadata);

    // Handle empty case
    if count == 0 {
        let output = CpuDevice::allocate(0, DType::I32)?;
        return Ok((output, 0));
    }

    // Allocate output buffer for [count, ndim] indices
    let output_size = count * ndim;
    let mut output = CpuDevice::allocate(output_size, DType::I32)?;

    // Second pass: fill indices
    let output_ptr = match &mut output {
        CpuStorage::I32(data) => data.as_mut_ptr(),
        _ => unreachable!("output should be I32"),
    };

    hodu_cpu_kernels::call_nonzero_fill(fill_kernel, input_ptr, output_ptr, &metadata)?;

    Ok((output, count))
}
