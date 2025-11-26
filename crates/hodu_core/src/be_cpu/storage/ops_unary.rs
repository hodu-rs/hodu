use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    compat::*,
    error::{HoduError, HoduResult},
    ops::Op,
    scalar::Scalar,
    types::{DType, Layout},
};
use core::ffi::c_void;

pub fn call_ops_cmp_scalar(
    input_storage: &CpuStorage,
    input_layout: &Layout,
    scalar: Scalar,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Extract cmp scalar op
    let cmp_op = match op {
        Op::CmpScalar(cmp_op) => cmp_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_cmp_scalar expects cmp scalar op".to_string(),
            ))
        },
    };

    let shape = input_layout.shape();
    let num_els = shape.size();

    // Compute output layout for metadata generation
    let output_layout = input_layout.clone(); // Cmp scalar ops preserve layout

    // Generate metadata using centralized function
    let metadata = crate::op_metadatas::cmp_scalar_metadata(input_layout, &output_layout);

    // Use Display to get kernel name
    let kernel_name = format!("{}_{}", cmp_op, input_storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage (cmp ops return BOOL)
    let mut output = CpuDevice::allocate(num_els, DType::BOOL)?;

    // Get raw pointers and call kernel with scalar
    macro_rules! call_kernel {
        ($in_data:expr, $out_data:expr, $scalar_val:expr) => {{
            let in_ptr = $in_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_unary_scalar(kernel, in_ptr, out_ptr, &metadata, $scalar_val)?;
        }};
    }

    match (input_storage, &mut output, scalar) {
        (CpuStorage::BOOL(inp), CpuStorage::BOOL(out), Scalar::BOOL(s)) => call_kernel!(inp, out, s),
        (CpuStorage::F8E4M3(inp), CpuStorage::BOOL(out), Scalar::F8E4M3(s)) => call_kernel!(inp, out, s),
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(inp), CpuStorage::BOOL(out), Scalar::F8E5M2(s)) => call_kernel!(inp, out, s),
        (CpuStorage::BF16(inp), CpuStorage::BOOL(out), Scalar::BF16(s)) => call_kernel!(inp, out, s),
        (CpuStorage::F16(inp), CpuStorage::BOOL(out), Scalar::F16(s)) => call_kernel!(inp, out, s),
        (CpuStorage::F32(inp), CpuStorage::BOOL(out), Scalar::F32(s)) => call_kernel!(inp, out, s),
        #[cfg(feature = "f64")]
        (CpuStorage::F64(inp), CpuStorage::BOOL(out), Scalar::F64(s)) => call_kernel!(inp, out, s),
        (CpuStorage::U8(inp), CpuStorage::BOOL(out), Scalar::U8(s)) => call_kernel!(inp, out, s),
        #[cfg(feature = "u16")]
        (CpuStorage::U16(inp), CpuStorage::BOOL(out), Scalar::U16(s)) => call_kernel!(inp, out, s),
        (CpuStorage::U32(inp), CpuStorage::BOOL(out), Scalar::U32(s)) => call_kernel!(inp, out, s),
        #[cfg(feature = "u64")]
        (CpuStorage::U64(inp), CpuStorage::BOOL(out), Scalar::U64(s)) => call_kernel!(inp, out, s),
        (CpuStorage::I8(inp), CpuStorage::BOOL(out), Scalar::I8(s)) => call_kernel!(inp, out, s),
        #[cfg(feature = "i16")]
        (CpuStorage::I16(inp), CpuStorage::BOOL(out), Scalar::I16(s)) => call_kernel!(inp, out, s),
        (CpuStorage::I32(inp), CpuStorage::BOOL(out), Scalar::I32(s)) => call_kernel!(inp, out, s),
        #[cfg(feature = "i64")]
        (CpuStorage::I64(inp), CpuStorage::BOOL(out), Scalar::I64(s)) => call_kernel!(inp, out, s),
        _ => {
            return Err(HoduError::BackendError(
                "mismatched storage/scalar types in call_cmp_scalar".to_string(),
            ))
        },
    }

    Ok(output)
}

pub fn call_ops_unary(input_storage: &CpuStorage, input_layout: &Layout, op: Op) -> HoduResult<CpuStorage> {
    // Extract unary op
    let unary_op = match op {
        Op::Unary(unary_op) => unary_op,
        _ => return Err(HoduError::BackendError("Lcall_unaryE expects LunaryE op".to_string())),
    };

    let shape = input_layout.shape();
    let num_els = shape.size();

    // Compute output layout for metadata generation
    let output_layout = input_layout.clone(); // Unary ops preserve layout

    // Generate metadata using centralized function
    let metadata = crate::op_metadatas::unary_metadata(input_layout, &output_layout);

    // Use Display to get kernel name
    let kernel_name = format!("{}_{}", unary_op, input_storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let dtype = input_storage.dtype();
    let mut output = CpuDevice::allocate(num_els, dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($in_data:expr, $out_data:expr) => {{
            let in_ptr = $in_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_unary(kernel, in_ptr, out_ptr, &metadata)?;
        }};
    }

    match (input_storage, &mut output) {
        (CpuStorage::BOOL(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
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
                "mismatched storage types in call_unary".to_string(),
            ))
        },
    }

    Ok(output)
}

pub fn call_ops_unary_logical(input_storage: &CpuStorage, input_layout: &Layout, op: Op) -> HoduResult<CpuStorage> {
    // Extract unary logical op
    let unary_op = match op {
        Op::UnaryLogical(unary_op) => unary_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_unary_logical expects unary logical op".to_string(),
            ))
        },
    };

    let shape = input_layout.shape();
    let num_els = shape.size();

    // Compute output layout for metadata generation
    let output_layout = input_layout.clone(); // Unary logical ops preserve layout

    // Generate metadata using centralized function
    let metadata = crate::op_metadatas::unary_logical_metadata(input_layout, &output_layout);

    // Use Display to get kernel name
    let kernel_name = format!("{}_{}", unary_op, input_storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage (logical ops return BOOL)
    let mut output = CpuDevice::allocate(num_els, DType::BOOL)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($in_data:expr, $out_data:expr) => {{
            let in_ptr = $in_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_unary(kernel, in_ptr, out_ptr, &metadata)?;
        }};
    }

    match (input_storage, &mut output) {
        (CpuStorage::BOOL(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
        (CpuStorage::F8E4M3(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
        (CpuStorage::BF16(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
        (CpuStorage::F16(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
        (CpuStorage::F32(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
        #[cfg(feature = "f64")]
        (CpuStorage::F64(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
        (CpuStorage::U8(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
        #[cfg(feature = "u16")]
        (CpuStorage::U16(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
        (CpuStorage::U32(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
        #[cfg(feature = "u64")]
        (CpuStorage::U64(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
        (CpuStorage::I8(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
        #[cfg(feature = "i16")]
        (CpuStorage::I16(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
        (CpuStorage::I32(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
        #[cfg(feature = "i64")]
        (CpuStorage::I64(inp), CpuStorage::BOOL(out)) => call_kernel!(inp, out),
        _ => {
            return Err(HoduError::BackendError(
                "mismatched storage types in call_unary_logical".to_string(),
            ))
        },
    }

    Ok(output)
}

pub fn call_ops_unary_scalar(
    input_storage: &CpuStorage,
    input_layout: &Layout,
    scalar: Scalar,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Extract unary scalar op
    let unary_op = match op {
        Op::UnaryScalar(unary_op) => unary_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_unary_scalar expects unary scalar op".to_string(),
            ))
        },
    };

    let shape = input_layout.shape();
    let num_els = shape.size();

    // Compute output layout for metadata generation
    let output_layout = input_layout.clone(); // Unary scalar ops preserve layout

    // Generate metadata using centralized function
    let metadata = crate::op_metadatas::unary_scalar_metadata(input_layout, &output_layout);

    // Use Display to get kernel name
    let kernel_name = format!("{}_{}", unary_op, input_storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let dtype = input_storage.dtype();
    let mut output = CpuDevice::allocate(num_els, dtype)?;

    // Get raw pointers and call kernel with scalar
    macro_rules! call_kernel {
        ($in_data:expr, $out_data:expr, $scalar_val:expr) => {{
            let in_ptr = $in_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_unary_scalar(kernel, in_ptr, out_ptr, &metadata, $scalar_val)?;
        }};
    }

    match (input_storage, &mut output, scalar) {
        (CpuStorage::BOOL(inp), CpuStorage::BOOL(out), Scalar::BOOL(s)) => call_kernel!(inp, out, s),
        (CpuStorage::F8E4M3(inp), CpuStorage::F8E4M3(out), Scalar::F8E4M3(s)) => call_kernel!(inp, out, s),
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(inp), CpuStorage::F8E5M2(out), Scalar::F8E5M2(s)) => call_kernel!(inp, out, s),
        (CpuStorage::BF16(inp), CpuStorage::BF16(out), Scalar::BF16(s)) => call_kernel!(inp, out, s),
        (CpuStorage::F16(inp), CpuStorage::F16(out), Scalar::F16(s)) => call_kernel!(inp, out, s),
        (CpuStorage::F32(inp), CpuStorage::F32(out), Scalar::F32(s)) => call_kernel!(inp, out, s),
        #[cfg(feature = "f64")]
        (CpuStorage::F64(inp), CpuStorage::F64(out), Scalar::F64(s)) => call_kernel!(inp, out, s),
        (CpuStorage::U8(inp), CpuStorage::U8(out), Scalar::U8(s)) => call_kernel!(inp, out, s),
        #[cfg(feature = "u16")]
        (CpuStorage::U16(inp), CpuStorage::U16(out), Scalar::U16(s)) => call_kernel!(inp, out, s),
        (CpuStorage::U32(inp), CpuStorage::U32(out), Scalar::U32(s)) => call_kernel!(inp, out, s),
        #[cfg(feature = "u64")]
        (CpuStorage::U64(inp), CpuStorage::U64(out), Scalar::U64(s)) => call_kernel!(inp, out, s),
        (CpuStorage::I8(inp), CpuStorage::I8(out), Scalar::I8(s)) => call_kernel!(inp, out, s),
        #[cfg(feature = "i16")]
        (CpuStorage::I16(inp), CpuStorage::I16(out), Scalar::I16(s)) => call_kernel!(inp, out, s),
        (CpuStorage::I32(inp), CpuStorage::I32(out), Scalar::I32(s)) => call_kernel!(inp, out, s),
        #[cfg(feature = "i64")]
        (CpuStorage::I64(inp), CpuStorage::I64(out), Scalar::I64(s)) => call_kernel!(inp, out, s),
        _ => {
            return Err(HoduError::BackendError(
                "mismatched storage/scalar types in call_unary_scalar".to_string(),
            ))
        },
    }

    Ok(output)
}
