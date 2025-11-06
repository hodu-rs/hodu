use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    types::{Layout, Shape},
};
use core::ffi::c_void;

/// Execute concat operation to concatenate multiple tensors along a dimension
///
/// Concatenates multiple input tensors along the specified dimension.
///
/// # Arguments
/// * `first` - First tensor storage
/// * `others` - Other tensor storages to concatenate
/// * `layouts` - All tensor layouts (including first)
/// * `dim` - Dimension along which to concatenate
/// * `op` - The concat operation (should be Op::Concat(ConcatOp))
///
/// # Returns
/// Output storage containing the concatenated tensor
pub fn call_concat(
    first: &CpuStorage,
    others: &[&CpuStorage],
    layouts: &[&Layout],
    dim: u32,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Collect all storages
    let mut storages = vec![first];
    storages.extend(others.iter().copied());
    // Validate op
    match op {
        Op::Concat(_) => (),
        _ => return Err(HoduError::BackendError("Lcall_concatE expects LconcatE op".to_string())),
    }

    if layouts.is_empty() {
        return Err(HoduError::BackendError(
            "concat requires at least one input".to_string(),
        ));
    }

    if storages.len() != layouts.len() {
        return Err(HoduError::BackendError(
            "storages and layouts length mismatch".to_string(),
        ));
    }

    let num_inputs = storages.len();
    let first_shape = layouts[0].shape();
    let ndim = first_shape.ndim();

    // Validate dim
    if dim >= ndim {
        return Err(HoduError::InvalidAxis { axis: dim as i32, ndim });
    }

    // Validate all inputs have same dtype and compatible shapes
    let dtype = storages[0].dtype();
    for (i, storage) in storages.iter().enumerate() {
        if storage.dtype() != dtype {
            return Err(HoduError::DTypeMismatch {
                expected: dtype,
                got: storage.dtype(),
            });
        }

        let shape = layouts[i].shape();
        if shape.ndim() != ndim {
            return Err(HoduError::BackendError(format!(
                "all inputs must have same number of dimensions, expected {}, got {}",
                ndim,
                shape.ndim()
            )));
        }

        // Check all dimensions except concat_dim match
        for d in 0..ndim {
            if d != dim && shape.dims()[d as usize] != first_shape.dims()[d as usize] {
                return Err(HoduError::BackendError(format!(
                    "dimension {} mismatch: expected {}, got {}",
                    d,
                    first_shape.dims()[d as usize],
                    shape.dims()[d as usize]
                )));
            }
        }
    }

    // Compute output shape
    let mut output_shape_vec = first_shape.dims().to_vec();
    output_shape_vec[dim as usize] = layouts.iter().map(|l| l.shape().dims()[dim as usize]).sum();
    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();

    // Build metadata array for CPU kernel
    // Layout: num_els, num_dims, output_shape, concat_dim, num_inputs,
    //         input_shapes (flattened), input_strides (flattened),
    //         input_offsets, input_buffer_offsets
    let mut metadata = Vec::with_capacity(
        2 + ndim as usize + 1 + 1 + num_inputs * ndim as usize + num_inputs * ndim as usize + num_inputs + num_inputs,
    );

    metadata.push(num_els as usize);
    metadata.push(ndim as usize);

    // Add output shape
    for &d in &output_shape_vec {
        metadata.push(d as usize);
    }

    // Add concat dimension
    metadata.push(dim as usize);

    // Add number of inputs
    metadata.push(num_inputs);

    // Add input shapes (flattened)
    for layout in layouts {
        for &d in layout.shape().dims() {
            metadata.push(d as usize);
        }
    }

    // Add input strides (flattened)
    for layout in layouts {
        for &s in layout.strides() {
            metadata.push(s as usize);
        }
    }

    // Add input offsets
    for layout in layouts {
        metadata.push(layout.offset() as usize);
    }

    // Add input buffer offsets (we'll pack all inputs contiguously)
    let mut buffer_offset = 0;
    for layout in layouts {
        metadata.push(buffer_offset);
        buffer_offset += layout.shape().size() as usize;
    }

    // Generate kernel name
    let kernel_name = format!("concat_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let mut output = CpuDevice::zeros(&output_shape, dtype)?;

    // Pack all inputs into a single buffer and call kernel based on dtype
    macro_rules! concat_impl {
        ($variant:ident, $inner_type:ty) => {{
            // Extract all data vectors
            let mut input_vecs = Vec::new();
            for storage in &storages {
                match storage {
                    CpuStorage::$variant(data) => input_vecs.push(data.as_slice()),
                    _ => {
                        return Err(HoduError::DTypeMismatch {
                            expected: dtype,
                            got: storage.dtype(),
                        })
                    },
                }
            }

            // Pack into contiguous buffer
            let mut input_buffer: Vec<$inner_type> = Vec::new();
            for data in input_vecs {
                input_buffer.extend_from_slice(data);
            }

            let input_ptr = input_buffer.as_ptr() as *const c_void;

            match &mut output {
                CpuStorage::$variant(out) => {
                    let out_ptr = out.as_mut_ptr() as *mut c_void;
                    hodu_cpu_kernels::concat_split::call_concat(kernel, input_ptr, out_ptr, &metadata)?;
                },
                _ => unreachable!(),
            }
        }};
    }

    match dtype {
        crate::types::DType::F8E4M3 => concat_impl!(F8E4M3, float8::F8E4M3),
        #[cfg(feature = "f8e5m2")]
        crate::types::DType::F8E5M2 => concat_impl!(F8E5M2, float8::F8E5M2),
        crate::types::DType::BF16 => concat_impl!(BF16, half::bf16),
        crate::types::DType::F16 => concat_impl!(F16, half::f16),
        crate::types::DType::F32 => concat_impl!(F32, f32),
        #[cfg(feature = "f64")]
        crate::types::DType::F64 => concat_impl!(F64, f64),
        crate::types::DType::U8 => concat_impl!(U8, u8),
        #[cfg(feature = "u16")]
        crate::types::DType::U16 => concat_impl!(U16, u16),
        crate::types::DType::U32 => concat_impl!(U32, u32),
        #[cfg(feature = "u64")]
        crate::types::DType::U64 => concat_impl!(U64, u64),
        crate::types::DType::I8 => concat_impl!(I8, i8),
        #[cfg(feature = "i16")]
        crate::types::DType::I16 => concat_impl!(I16, i16),
        crate::types::DType::I32 => concat_impl!(I32, i32),
        #[cfg(feature = "i64")]
        crate::types::DType::I64 => concat_impl!(I64, i64),
        crate::types::DType::BOOL => concat_impl!(BOOL, bool),
    }

    Ok(output)
}

/// Execute split operation to extract a slice from a tensor
///
/// Extracts a slice from the input tensor along the specified dimension.
///
/// # Arguments
/// * `storage` - Input tensor storage
/// * `layout` - Input tensor layout
/// * `dim` - Dimension along which to split
/// * `start` - Starting index along the split dimension
/// * `size` - Size of the slice along the split dimension
/// * `op` - The split operation (should be Op::Split(SplitOp))
///
/// # Returns
/// Output storage containing the extracted slice
pub fn call_split(
    storage: &CpuStorage,
    layout: &Layout,
    dim: u32,
    start: u32,
    size: u32,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Validate op
    match op {
        Op::Split(_) => (),
        _ => return Err(HoduError::BackendError("Lcall_splitE expects LsplitE op".to_string())),
    }

    let input_shape = layout.shape();
    let ndim = input_shape.ndim();

    // Validate dim
    if dim >= ndim {
        return Err(HoduError::InvalidAxis { axis: dim as i32, ndim });
    }

    // Validate start and size
    let dim_size = input_shape.dims()[dim as usize];
    if start >= dim_size {
        return Err(HoduError::BackendError(format!(
            "split start {} exceeds dimension size {}",
            start, dim_size
        )));
    }
    if start + size > dim_size {
        return Err(HoduError::BackendError(format!(
            "split range {}..{} exceeds dimension size {}",
            start,
            start + size,
            dim_size
        )));
    }

    // Compute output shape
    let mut output_shape_vec = input_shape.dims().to_vec();
    output_shape_vec[dim as usize] = size;
    let output_shape = Shape::new(&output_shape_vec);
    let num_els = output_shape.size();

    // Build metadata array for CPU kernel
    // Layout: num_els, num_dims, input_shape, input_strides, input_offset,
    //         split_dim, output_size_on_dim, split_offset
    let mut metadata = Vec::with_capacity(2 + ndim as usize + ndim as usize + 1 + 3);

    metadata.push(num_els as usize);
    metadata.push(ndim as usize);

    // Add input shape
    for &d in input_shape.dims() {
        metadata.push(d as usize);
    }

    // Add input strides
    for &s in layout.strides() {
        metadata.push(s as usize);
    }

    // Add input offset
    metadata.push(layout.offset() as usize);

    // Add split dimension
    metadata.push(dim as usize);

    // Add output size on split dimension
    metadata.push(size as usize);

    // Add split offset (start position)
    metadata.push(start as usize);

    // Generate kernel name
    let dtype = storage.dtype();
    let kernel_name = format!("split_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let mut output = CpuDevice::zeros(&output_shape, dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($input_data:expr, $out_data:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::concat_split::call_split(kernel, input_ptr, out_ptr, &metadata)?;
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
                "mismatched storage types in call_split".to_string(),
            ))
        },
    }

    Ok(output)
}
