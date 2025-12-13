use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    error::{HoduError, HoduResult},
    types::{Layout, Shape},
};
use core::ffi::c_void;

/// Execute determinant operation for square matrices
///
/// Computes the determinant of square matrices with optional batch dimensions.
/// Input: [..., N, N] -> Output: [...]
///
/// # Arguments
/// * `storage` - Input tensor storage (square matrix)
/// * `layout` - Layout of input tensor
///
/// # Returns
/// Output storage containing the determinant(s)
pub fn call_ops_det(storage: &CpuStorage, layout: &Layout) -> HoduResult<CpuStorage> {
    let shape = layout.shape();
    let ndim = shape.ndim();

    // Validate shape
    if ndim < 2 {
        return Err(HoduError::BackendError("det requires at least 2D tensor".to_string()));
    }

    let n = shape.dims()[ndim - 1];
    let m = shape.dims()[ndim - 2];

    if n != m {
        return Err(HoduError::BackendError(format!(
            "det requires square matrix, got {}×{}",
            m, n
        )));
    }

    // Compute output shape (batch dimensions only)
    let output_shape = if ndim == 2 {
        Shape::new(&[1])
    } else {
        Shape::new(&shape.dims()[..ndim - 2])
    };

    // Generate metadata
    let metadata = crate::op_metadatas::det_metadata(layout)?;

    // Generate kernel name
    let kernel_name = format!("hodu_cpu_det_{}", storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let dtype = storage.dtype();
    let mut output = CpuDevice::allocate(output_shape.size(), dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($in_data:expr, $out_data:expr) => {{
            let in_ptr = $in_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_det(kernel, in_ptr, out_ptr, &metadata)?;
        }};
    }

    match (storage, &mut output) {
        (CpuStorage::F8E4M3(inp), CpuStorage::F8E4M3(out)) => call_kernel!(inp, out),
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(inp), CpuStorage::F8E5M2(out)) => call_kernel!(inp, out),
        (CpuStorage::BF16(inp), CpuStorage::BF16(out)) => call_kernel!(inp, out),
        (CpuStorage::F16(inp), CpuStorage::F16(out)) => call_kernel!(inp, out),
        (CpuStorage::F32(inp), CpuStorage::F32(out)) => call_kernel!(inp, out),
        #[cfg(feature = "f64")]
        (CpuStorage::F64(inp), CpuStorage::F64(out)) => call_kernel!(inp, out),
        (CpuStorage::U8(inp), CpuStorage::U8(out)) => call_kernel!(inp, out),
        #[cfg(feature = "u16")]
        (CpuStorage::U16(inp), CpuStorage::U16(out)) => call_kernel!(inp, out),
        (CpuStorage::U32(inp), CpuStorage::U32(out)) => call_kernel!(inp, out),
        #[cfg(feature = "u64")]
        (CpuStorage::U64(inp), CpuStorage::U64(out)) => call_kernel!(inp, out),
        (CpuStorage::I8(inp), CpuStorage::I8(out)) => call_kernel!(inp, out),
        #[cfg(feature = "i16")]
        (CpuStorage::I16(inp), CpuStorage::I16(out)) => call_kernel!(inp, out),
        (CpuStorage::I32(inp), CpuStorage::I32(out)) => call_kernel!(inp, out),
        #[cfg(feature = "i64")]
        (CpuStorage::I64(inp), CpuStorage::I64(out)) => call_kernel!(inp, out),
        _ => {
            return Err(HoduError::BackendError(
                "mismatched storage types in call_det".to_string(),
            ))
        },
    }

    Ok(output)
}

/// Execute matrix inverse operation for square matrices
///
/// Computes the inverse of square matrices with optional batch dimensions.
/// Input: [..., N, N] -> Output: [..., N, N]
///
/// # Arguments
/// * `storage` - Input tensor storage (square matrix)
/// * `layout` - Layout of input tensor
///
/// # Returns
/// Output storage containing the inverse matrix/matrices
pub fn call_ops_inv(storage: &CpuStorage, layout: &Layout) -> HoduResult<CpuStorage> {
    let shape = layout.shape();
    let ndim = shape.ndim();

    // Validate shape
    if ndim < 2 {
        return Err(HoduError::BackendError("inv requires at least 2D tensor".to_string()));
    }

    let n = shape.dims()[ndim - 1];
    let m = shape.dims()[ndim - 2];

    if n != m {
        return Err(HoduError::BackendError(format!(
            "inv requires square matrix, got {}×{}",
            m, n
        )));
    }

    // Output shape is same as input (inverse has same shape)
    let output_shape = shape.clone();

    // Generate metadata (same as det)
    let metadata = crate::op_metadatas::inv_metadata(layout)?;

    // Generate kernel name
    let kernel_name = format!("hodu_cpu_inv_{}", storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let dtype = storage.dtype();
    let mut output = CpuDevice::allocate(output_shape.size(), dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($in_data:expr, $out_data:expr) => {{
            let in_ptr = $in_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_inv(kernel, in_ptr, out_ptr, &metadata)?;
        }};
    }

    match (storage, &mut output) {
        (CpuStorage::F8E4M3(inp), CpuStorage::F8E4M3(out)) => call_kernel!(inp, out),
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(inp), CpuStorage::F8E5M2(out)) => call_kernel!(inp, out),
        (CpuStorage::BF16(inp), CpuStorage::BF16(out)) => call_kernel!(inp, out),
        (CpuStorage::F16(inp), CpuStorage::F16(out)) => call_kernel!(inp, out),
        (CpuStorage::F32(inp), CpuStorage::F32(out)) => call_kernel!(inp, out),
        #[cfg(feature = "f64")]
        (CpuStorage::F64(inp), CpuStorage::F64(out)) => call_kernel!(inp, out),
        (CpuStorage::U8(inp), CpuStorage::U8(out)) => call_kernel!(inp, out),
        #[cfg(feature = "u16")]
        (CpuStorage::U16(inp), CpuStorage::U16(out)) => call_kernel!(inp, out),
        (CpuStorage::U32(inp), CpuStorage::U32(out)) => call_kernel!(inp, out),
        #[cfg(feature = "u64")]
        (CpuStorage::U64(inp), CpuStorage::U64(out)) => call_kernel!(inp, out),
        (CpuStorage::I8(inp), CpuStorage::I8(out)) => call_kernel!(inp, out),
        #[cfg(feature = "i16")]
        (CpuStorage::I16(inp), CpuStorage::I16(out)) => call_kernel!(inp, out),
        (CpuStorage::I32(inp), CpuStorage::I32(out)) => call_kernel!(inp, out),
        #[cfg(feature = "i64")]
        (CpuStorage::I64(inp), CpuStorage::I64(out)) => call_kernel!(inp, out),
        _ => {
            return Err(HoduError::BackendError(
                "mismatched storage types in call_inv".to_string(),
            ))
        },
    }

    Ok(output)
}

/// Execute matrix trace operation for square matrices
///
/// Computes the trace (sum of diagonal elements) of square matrices with optional batch dimensions.
/// Input: [..., N, N] -> Output: [...]
///
/// # Arguments
/// * `storage` - Input tensor storage (square matrix)
/// * `layout` - Layout of input tensor
///
/// # Returns
/// Output storage containing the trace(s)
pub fn call_ops_trace(storage: &CpuStorage, layout: &Layout) -> HoduResult<CpuStorage> {
    let shape = layout.shape();
    let ndim = shape.ndim();

    // Validate shape
    if ndim < 2 {
        return Err(HoduError::BackendError("trace requires at least 2D tensor".to_string()));
    }

    let n = shape.dims()[ndim - 1];
    let m = shape.dims()[ndim - 2];

    if n != m {
        return Err(HoduError::BackendError(format!(
            "trace requires square matrix, got {}×{}",
            m, n
        )));
    }

    // Compute output shape (batch dimensions only)
    let output_shape = if ndim == 2 {
        Shape::new(&[1])
    } else {
        Shape::new(&shape.dims()[..ndim - 2])
    };

    // Generate metadata (same as det/inv)
    let metadata = crate::op_metadatas::trace_metadata(layout)?;

    // Generate kernel name
    let kernel_name = format!("hodu_cpu_trace_{}", storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let dtype = storage.dtype();
    let mut output = CpuDevice::allocate(output_shape.size(), dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($in_data:expr, $out_data:expr) => {{
            let in_ptr = $in_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_trace(kernel, in_ptr, out_ptr, &metadata)?;
        }};
    }

    match (storage, &mut output) {
        (CpuStorage::F8E4M3(inp), CpuStorage::F8E4M3(out)) => call_kernel!(inp, out),
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(inp), CpuStorage::F8E5M2(out)) => call_kernel!(inp, out),
        (CpuStorage::BF16(inp), CpuStorage::BF16(out)) => call_kernel!(inp, out),
        (CpuStorage::F16(inp), CpuStorage::F16(out)) => call_kernel!(inp, out),
        (CpuStorage::F32(inp), CpuStorage::F32(out)) => call_kernel!(inp, out),
        #[cfg(feature = "f64")]
        (CpuStorage::F64(inp), CpuStorage::F64(out)) => call_kernel!(inp, out),
        (CpuStorage::U8(inp), CpuStorage::U8(out)) => call_kernel!(inp, out),
        #[cfg(feature = "u16")]
        (CpuStorage::U16(inp), CpuStorage::U16(out)) => call_kernel!(inp, out),
        (CpuStorage::U32(inp), CpuStorage::U32(out)) => call_kernel!(inp, out),
        #[cfg(feature = "u64")]
        (CpuStorage::U64(inp), CpuStorage::U64(out)) => call_kernel!(inp, out),
        (CpuStorage::I8(inp), CpuStorage::I8(out)) => call_kernel!(inp, out),
        #[cfg(feature = "i16")]
        (CpuStorage::I16(inp), CpuStorage::I16(out)) => call_kernel!(inp, out),
        (CpuStorage::I32(inp), CpuStorage::I32(out)) => call_kernel!(inp, out),
        #[cfg(feature = "i64")]
        (CpuStorage::I64(inp), CpuStorage::I64(out)) => call_kernel!(inp, out),
        _ => {
            return Err(HoduError::BackendError(
                "mismatched storage types in call_trace".to_string(),
            ))
        },
    }

    Ok(output)
}
