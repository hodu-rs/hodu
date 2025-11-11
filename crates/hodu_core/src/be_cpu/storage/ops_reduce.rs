use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    types::{Layout, Shape},
};
use core::ffi::c_void;
use smallvec::SmallVec;

/// Execute reduce operation along specified dimensions
///
/// Reduces input tensor along the given dimensions using operations like sum, mean, max, min, etc.
///
/// # Arguments
/// * `storage` - Input tensor storage
/// * `layout` - Layout of input tensor
/// * `dims` - Dimensions to reduce over
/// * `keep_dim` - Whether to keep reduced dimensions as size 1
/// * `op` - The reduce operation (should be Op::Reduce(ReduceOp::...))
///
/// # Returns
/// Output storage containing the reduced tensor
pub fn call_ops_reduce(
    storage: &CpuStorage,
    layout: &Layout,
    dims: &[u32],
    keep_dim: bool,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Extract reduce op
    let reduce_op = match op {
        Op::Reduce(reduce_op) => reduce_op,
        _ => return Err(HoduError::BackendError("Lcall_reduceE expects LreduceE op".to_string())),
    };

    let input_shape = layout.shape();
    let input_ndim = input_shape.ndim();

    // Validate reduce dimensions
    for &dim in dims {
        if dim >= input_ndim {
            return Err(HoduError::InvalidAxis {
                axis: dim as i32,
                ndim: input_ndim,
            });
        }
    }

    // Compute output shape
    let mut output_shape_vec = SmallVec::<[u32; 24]>::new();
    for i in 0..input_ndim {
        if dims.contains(&i) {
            if keep_dim {
                output_shape_vec.push(1);
            }
        } else {
            output_shape_vec.push(input_shape.dims()[i as usize]);
        }
    }

    // Handle empty output shape (reduce all dimensions without keep_dim)
    if output_shape_vec.is_empty() {
        output_shape_vec.push(1);
    }

    let output_shape = Shape::new(&output_shape_vec);

    // Calculate reduce size (number of elements to reduce per output element)
    let mut reduce_size: u64 = 1;
    for &dim in dims {
        reduce_size *= input_shape.dims()[dim as usize] as u64;
    }

    // Build metadata array for CPU kernel
    // Layout: shape_len, shape, strides, offset, output_shape_len, output_shape,
    //         num_reduce_dims, reduce_dims, keep_dim, reduce_size
    let mut metadata: SmallVec<[usize; 24]> = SmallVec::with_capacity(
        1 + input_ndim as usize + input_ndim as usize + 1 + 1 + output_shape_vec.len() + 1 + dims.len() + 1 + 1,
    );

    // Add input shape info
    metadata.push(input_ndim as usize);
    for &dim in input_shape.dims() {
        metadata.push(dim as usize);
    }

    // Add input strides
    for &stride in layout.strides() {
        metadata.push(stride as usize);
    }

    // Add input offset
    metadata.push(layout.offset() as usize);

    // Add output shape info
    metadata.push(output_shape_vec.len());
    for &dim in &output_shape_vec {
        metadata.push(dim as usize);
    }

    // Add reduce dimensions
    metadata.push(dims.len());
    for &dim in dims {
        metadata.push(dim as usize);
    }

    // Add keep_dim flag
    metadata.push(if keep_dim { 1 } else { 0 });

    // Add reduce size
    metadata.push(reduce_size as usize);

    // Generate kernel name
    let kernel_name = format!("{}_{}", reduce_op, storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let dtype = storage.dtype();
    let mut output = CpuDevice::zeros(&output_shape, dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($input_data:expr, $out_data:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_reduce(kernel, input_ptr, out_ptr, &metadata)?;
        }};
    }

    match (storage, &mut output) {
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
                "mismatched storage types in call_reduce".to_string(),
            ))
        },
    }

    Ok(output)
}
