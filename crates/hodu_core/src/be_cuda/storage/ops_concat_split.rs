use crate::{
    be::storage::BackendStorageT,
    be_cuda::storage::{CudaStorage, CudaStorageData},
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    types::{Layout, Shape},
};
use hodu_cuda_kernels::kernels;

#[allow(clippy::needless_range_loop)]
pub fn call_ops_concat(
    first: &CudaStorage,
    others: &[&CudaStorage],
    layouts: &[&Layout],
    dim: u32,
    op: Op,
) -> HoduResult<CudaStorage> {
    // Collect all storages
    let mut storages = vec![first];
    storages.extend(others.iter().copied());

    // Validate op
    match op {
        Op::Concat(_) => (),
        _ => return Err(HoduError::BackendError("call_ops_concat expects concat op".to_string())),
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

    let device = first.get_device();

    // Strategy: Download to CPU, pack into single buffer, upload back to GPU
    // This is simpler than device-to-device copies and works for now

    // Build metadata for concat kernel
    let mut metadata = Vec::new();
    metadata.push(num_els as usize);
    metadata.push(ndim as usize);

    // Add output shape
    for &d in &output_shape_vec {
        metadata.push(d as usize);
    }

    metadata.push(dim as usize);
    metadata.push(num_inputs);

    // Add input shapes
    for layout in layouts {
        for &d in layout.shape().dims() {
            metadata.push(d as usize);
        }
    }

    // Add input strides
    for layout in layouts {
        for &s in layout.strides() {
            metadata.push(s as usize);
        }
    }

    // Add input offsets
    for layout in layouts {
        metadata.push(layout.offset() as usize);
    }

    // Add input buffer offsets - cumulative positions in packed temp buffer (in elements)
    let mut buffer_offset_elements = 0;
    let mut buffer_offsets = Vec::new();
    for storage in storages.iter() {
        buffer_offsets.push(buffer_offset_elements);
        buffer_offset_elements += storage.len();
    }
    metadata.extend(&buffer_offsets);

    let kernel_name = format!("concat_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Implement concat for each dtype
    macro_rules! impl_concat {
        ($ty:ty, $variant:ident) => {{
            // Download all storages to CPU and pack into single vector
            let mut packed_data = Vec::new();
            for storage in storages.iter() {
                if let CudaStorageData::$variant(slice) = &storage.data {
                    let stream = device.context().default_stream();
                    let mut temp = vec![unsafe { core::mem::zeroed() }; slice.len()];
                    stream
                        .memcpy_dtoh(slice, &mut temp)
                        .map_err(|e| HoduError::BackendError(format!("CUDA memcpy_dtoh failed: {:?}", e)))?;
                    packed_data.extend_from_slice(&temp);
                } else {
                    return Err(HoduError::BackendError("dtype mismatch in concat".to_string()));
                }
            }

            // Upload packed data to GPU
            let temp_input_buffer = device.new_buffer_with_data(&packed_data)?;

            // Create output buffer
            let mut output = device.new_buffer::<$ty>(num_els as usize)?;

            // Call concat kernel
            kernels::call_ops_concat(
                kernel,
                device.kernels(),
                device.context(),
                &temp_input_buffer,
                &mut output,
                &metadata,
            )?;

            Ok(CudaStorage::new(
                first.device_id(),
                device.clone(),
                CudaStorageData::$variant(output),
            ))
        }};
    }

    match dtype {
        crate::types::DType::F32 => impl_concat!(f32, F32),
        crate::types::DType::I32 => impl_concat!(i32, I32),
        crate::types::DType::U32 => impl_concat!(u32, U32),
        crate::types::DType::F16 => impl_concat!(half::f16, F16),
        crate::types::DType::BF16 => impl_concat!(half::bf16, BF16),
        #[cfg(feature = "f64")]
        crate::types::DType::F64 => impl_concat!(f64, F64),
        _ => Err(HoduError::NotImplemented(format!("concat for dtype {:?}", dtype))),
    }
}

pub fn call_ops_split(
    storage: &CudaStorage,
    layout: &Layout,
    dim: u32,
    start: u32,
    size: u32,
    op: Op,
) -> HoduResult<CudaStorage> {
    // Validate op
    match op {
        Op::Split(_) => (),
        _ => return Err(HoduError::BackendError("call_ops_split expects split op".to_string())),
    }

    let input_shape = layout.shape();
    let ndim = input_shape.ndim();

    // Validate dim
    if dim >= ndim {
        return Err(HoduError::InvalidAxis { axis: dim as i32, ndim });
    }

    let dim_size = input_shape.dims()[dim as usize];
    if start + size > dim_size {
        return Err(HoduError::BackendError(format!(
            "split range [{}, {}) exceeds dimension size {}",
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

    // Build metadata array for CUDA kernel
    let mut metadata = Vec::new();
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

    metadata.push(layout.offset() as usize);

    // Add output shape
    for &d in &output_shape_vec {
        metadata.push(d as usize);
    }

    metadata.push(dim as usize);
    metadata.push(start as usize);

    let dtype = storage.dtype();
    let device = storage.get_device();

    // Get kernel name
    let kernel_name = format!("split_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    macro_rules! impl_split {
        ($input:expr, $ty:ty, $variant:ident) => {{
            let mut output = device.new_buffer::<$ty>(num_els as usize)?;
            kernels::call_ops_split(
                kernel,
                device.kernels(),
                device.context(),
                $input,
                &mut output,
                &metadata,
            )?;
            Ok(CudaStorage::new(
                storage.device_id(),
                device.clone(),
                CudaStorageData::$variant(output),
            ))
        }};
    }

    match &storage.data {
        CudaStorageData::F32(input) => impl_split!(input, f32, F32),
        CudaStorageData::I32(input) => impl_split!(input, i32, I32),
        CudaStorageData::U32(input) => impl_split!(input, u32, U32),
        CudaStorageData::F16(input) => impl_split!(input, half::f16, F16),
        CudaStorageData::BF16(input) => impl_split!(input, half::bf16, BF16),
        #[cfg(feature = "f64")]
        CudaStorageData::F64(input) => impl_split!(input, f64, F64),
        _ => Err(HoduError::NotImplemented(format!("split for dtype {:?}", dtype))),
    }
}
