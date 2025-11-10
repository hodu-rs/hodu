use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    types::{Layout, Shape},
};
use core::ffi::c_void;

/// Execute reduce window operation
///
/// Applies a reduction operation over sliding windows of the input tensor.
///
/// # Arguments
/// * `storage` - Input tensor storage
/// * `layout` - Input tensor layout
/// * `window_shape` - Size of the window in each dimension
/// * `strides` - Step size in each dimension
/// * `padding` - Padding before and after for each dimension (flattened: [pad_before_0, pad_after_0, pad_before_1, pad_after_1, ...])
/// * `reduce_op` - The reduction operation to apply (Sum, Mean, Max, Min)
/// * `op` - The windowing operation (should be Op::Windowing(WindowingOp::ReduceWindow))
///
/// # Returns
/// Output storage containing the reduced windows
pub fn call_ops_reduce_window(
    storage: &CpuStorage,
    layout: &Layout,
    window_shape: &[u32],
    strides: &[u32],
    padding: &[u32],
    op: Op,
) -> HoduResult<CpuStorage> {
    // Extract windowing op
    let windowing_op = match op {
        Op::Windowing(windowing_op) => windowing_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_reduce_window expects Windowing op".to_string(),
            ))
        },
    };

    // Get kernel prefix from windowing op display
    let kernel_prefix = format!("{}", windowing_op);

    let input_shape = layout.shape();
    let ndim = input_shape.ndim() as usize;

    // Validate window_shape, strides, padding dimensions
    if window_shape.len() != ndim {
        return Err(HoduError::BackendError(format!(
            "window_shape length {} does not match tensor ndim {}",
            window_shape.len(),
            ndim
        )));
    }

    if strides.len() != ndim {
        return Err(HoduError::BackendError(format!(
            "strides length {} does not match tensor ndim {}",
            strides.len(),
            ndim
        )));
    }

    if padding.len() != ndim * 2 {
        return Err(HoduError::BackendError(format!(
            "padding length {} does not match tensor ndim * 2 ({})",
            padding.len(),
            ndim * 2
        )));
    }

    // Compute output shape
    let mut output_shape_vec = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let in_size = input_shape.dims()[i];
        let window_size = window_shape[i];
        let stride = strides[i];
        let pad_before = padding[i * 2];
        let pad_after = padding[i * 2 + 1];

        // Output size formula: floor((in_size + pad_before + pad_after - window_size) / stride) + 1
        let padded_size = in_size + pad_before + pad_after;
        if padded_size < window_size {
            return Err(HoduError::BackendError(format!(
                "padded size {} is less than window size {} in dimension {}",
                padded_size, window_size, i
            )));
        }

        let out_size = (padded_size - window_size) / stride + 1;
        output_shape_vec.push(out_size);
    }

    let output_shape = Shape::new(&output_shape_vec);
    let output_size = output_shape.size();

    // Build metadata array
    // Layout: output_size, num_dims, input_shape, input_strides, offset,
    //         window_shape, strides, padding, output_shape
    let mut metadata = Vec::with_capacity(3 + ndim * 7);

    metadata.push(output_size as usize);
    metadata.push(ndim);

    // Add input shape
    for &dim in input_shape.dims() {
        metadata.push(dim as usize);
    }

    // Add input strides
    for &stride in layout.strides() {
        metadata.push(stride as usize);
    }

    // Add offset
    metadata.push(layout.offset() as usize);

    // Add window shape
    for &w in window_shape {
        metadata.push(w as usize);
    }

    // Add strides
    for &s in strides {
        metadata.push(s as usize);
    }

    // Add padding (already flattened)
    for &p in padding {
        metadata.push(p as usize);
    }

    // Add output shape
    for &dim in &output_shape_vec {
        metadata.push(dim as usize);
    }

    // Generate kernel name
    let dtype = storage.dtype();
    let kernel_name = format!("{}_{}", kernel_prefix, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let mut output = CpuDevice::zeros(&output_shape, dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($input_data:expr, $out_data:expr) => {{
            let input_ptr = $input_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_reduce_window(kernel, input_ptr, out_ptr, &metadata)?;
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
                "mismatched storage types in call_reduce_window".to_string(),
            ))
        },
    }

    Ok(output)
}
