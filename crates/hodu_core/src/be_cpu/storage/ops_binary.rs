use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::Op,
    types::{Layout, Shape},
};
use core::ffi::c_void;

pub fn call_binary(
    lhs_storage: &CpuStorage,
    rhs_storage: &CpuStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Extract binary op
    let binary_op = match op {
        Op::Binary(binary_op) => binary_op,
        _ => return Err(HoduError::InternalError("call_binary expects binary op".to_string())),
    };

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let num_els = lhs_shape.size();
    let num_dims = lhs_shape.ndim();

    // Build metadata array for CPU kernel
    let mut metadata = Vec::with_capacity(2 + 4 * num_dims as usize + 2);
    metadata.push(num_els as usize);
    metadata.push(num_dims as usize);

    // Add shapes
    for &dim in lhs_shape.dims() {
        metadata.push(dim as usize);
    }
    for &dim in rhs_shape.dims() {
        metadata.push(dim as usize);
    }

    // Add strides
    for &stride in lhs_layout.strides() {
        metadata.push(stride as usize);
    }
    for &stride in rhs_layout.strides() {
        metadata.push(stride as usize);
    }

    // Add offsets
    metadata.push(lhs_layout.offset() as usize);
    metadata.push(rhs_layout.offset() as usize);

    // Use Display to get kernel name
    let kernel_name = format!("{}_{}", binary_op, lhs_storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let dtype = lhs_storage.dtype();
    let mut output = CpuDevice::zeros(&Shape::new(&[num_els]), dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($lhs_data:expr, $rhs_data:expr, $out_data:expr) => {{
            let lhs_ptr = $lhs_data.as_ptr() as *const c_void;
            let rhs_ptr = $rhs_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::binary::call_binary(kernel, lhs_ptr, rhs_ptr, out_ptr, &metadata)?;
        }};
    }

    match (lhs_storage, rhs_storage, &mut output) {
        (CpuStorage::BOOL(lhs), CpuStorage::BOOL(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::F8E4M3(lhs), CpuStorage::F8E4M3(rhs), CpuStorage::F8E4M3(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(lhs), CpuStorage::F8E5M2(rhs), CpuStorage::F8E5M2(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::BF16(lhs), CpuStorage::BF16(rhs), CpuStorage::BF16(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::F16(lhs), CpuStorage::F16(rhs), CpuStorage::F16(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::F32(lhs), CpuStorage::F32(rhs), CpuStorage::F32(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "f64")]
        (CpuStorage::F64(lhs), CpuStorage::F64(rhs), CpuStorage::F64(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::U8(lhs), CpuStorage::U8(rhs), CpuStorage::U8(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "u16")]
        (CpuStorage::U16(lhs), CpuStorage::U16(rhs), CpuStorage::U16(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::U32(lhs), CpuStorage::U32(rhs), CpuStorage::U32(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "u64")]
        (CpuStorage::U64(lhs), CpuStorage::U64(rhs), CpuStorage::U64(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::I8(lhs), CpuStorage::I8(rhs), CpuStorage::I8(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "i16")]
        (CpuStorage::I16(lhs), CpuStorage::I16(rhs), CpuStorage::I16(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::I32(lhs), CpuStorage::I32(rhs), CpuStorage::I32(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "i64")]
        (CpuStorage::I64(lhs), CpuStorage::I64(rhs), CpuStorage::I64(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        _ => {
            return Err(HoduError::InternalError(
                "mismatched storage types in call_binary".to_string(),
            ))
        },
    }

    Ok(output)
}

pub fn call_binary_logical(
    lhs_storage: &CpuStorage,
    rhs_storage: &CpuStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Extract binary logical op
    let binary_op = match op {
        Op::BinaryLogical(binary_op) => binary_op,
        _ => {
            return Err(HoduError::InternalError(
                "call_binary_logical expects binary logical op".to_string(),
            ))
        },
    };

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let num_els = lhs_shape.size();
    let num_dims = lhs_shape.ndim();

    // Build metadata array for CPU kernel
    let mut metadata = Vec::with_capacity(2 + 4 * num_dims as usize + 2);
    metadata.push(num_els as usize);
    metadata.push(num_dims as usize);

    // Add shapes
    for &dim in lhs_shape.dims() {
        metadata.push(dim as usize);
    }
    for &dim in rhs_shape.dims() {
        metadata.push(dim as usize);
    }

    // Add strides
    for &stride in lhs_layout.strides() {
        metadata.push(stride as usize);
    }
    for &stride in rhs_layout.strides() {
        metadata.push(stride as usize);
    }

    // Add offsets
    metadata.push(lhs_layout.offset() as usize);
    metadata.push(rhs_layout.offset() as usize);

    // Use Display to get kernel name
    let kernel_name = format!("{}_{}", binary_op, lhs_storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage (logical ops return BOOL)
    let mut output = CpuDevice::zeros(&Shape::new(&[num_els]), crate::types::DType::BOOL)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($lhs_data:expr, $rhs_data:expr, $out_data:expr) => {{
            let lhs_ptr = $lhs_data.as_ptr() as *const c_void;
            let rhs_ptr = $rhs_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::binary::call_binary(kernel, lhs_ptr, rhs_ptr, out_ptr, &metadata)?;
        }};
    }

    match (lhs_storage, rhs_storage, &mut output) {
        (CpuStorage::BOOL(lhs), CpuStorage::BOOL(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::F8E4M3(lhs), CpuStorage::F8E4M3(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(lhs), CpuStorage::F8E5M2(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::BF16(lhs), CpuStorage::BF16(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::F16(lhs), CpuStorage::F16(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::F32(lhs), CpuStorage::F32(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "f64")]
        (CpuStorage::F64(lhs), CpuStorage::F64(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::U8(lhs), CpuStorage::U8(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "u16")]
        (CpuStorage::U16(lhs), CpuStorage::U16(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::U32(lhs), CpuStorage::U32(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "u64")]
        (CpuStorage::U64(lhs), CpuStorage::U64(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::I8(lhs), CpuStorage::I8(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "i16")]
        (CpuStorage::I16(lhs), CpuStorage::I16(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::I32(lhs), CpuStorage::I32(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "i64")]
        (CpuStorage::I64(lhs), CpuStorage::I64(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        _ => {
            return Err(HoduError::InternalError(
                "mismatched storage types in call_binary_logical".to_string(),
            ))
        },
    }

    Ok(output)
}

pub fn call_cmp(
    lhs_storage: &CpuStorage,
    rhs_storage: &CpuStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Extract cmp op
    let cmp_op = match op {
        Op::Cmp(cmp_op) => cmp_op,
        _ => return Err(HoduError::InternalError("call_cmp expects cmp op".to_string())),
    };

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let num_els = lhs_shape.size();
    let num_dims = lhs_shape.ndim();

    // Build metadata array for CPU kernel
    let mut metadata = Vec::with_capacity(2 + 4 * num_dims as usize + 2);
    metadata.push(num_els as usize);
    metadata.push(num_dims as usize);

    // Add shapes
    for &dim in lhs_shape.dims() {
        metadata.push(dim as usize);
    }
    for &dim in rhs_shape.dims() {
        metadata.push(dim as usize);
    }

    // Add strides
    for &stride in lhs_layout.strides() {
        metadata.push(stride as usize);
    }
    for &stride in rhs_layout.strides() {
        metadata.push(stride as usize);
    }

    // Add offsets
    metadata.push(lhs_layout.offset() as usize);
    metadata.push(rhs_layout.offset() as usize);

    // Use Display to get kernel name
    let kernel_name = format!("{}_{}", cmp_op, lhs_storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage (cmp ops return BOOL)
    let mut output = CpuDevice::zeros(&Shape::new(&[num_els]), crate::types::DType::BOOL)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($lhs_data:expr, $rhs_data:expr, $out_data:expr) => {{
            let lhs_ptr = $lhs_data.as_ptr() as *const c_void;
            let rhs_ptr = $rhs_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::binary::call_binary(kernel, lhs_ptr, rhs_ptr, out_ptr, &metadata)?;
        }};
    }

    match (lhs_storage, rhs_storage, &mut output) {
        (CpuStorage::BOOL(lhs), CpuStorage::BOOL(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::F8E4M3(lhs), CpuStorage::F8E4M3(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "f8e5m2")]
        (CpuStorage::F8E5M2(lhs), CpuStorage::F8E5M2(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::BF16(lhs), CpuStorage::BF16(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::F16(lhs), CpuStorage::F16(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::F32(lhs), CpuStorage::F32(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "f64")]
        (CpuStorage::F64(lhs), CpuStorage::F64(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::U8(lhs), CpuStorage::U8(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "u16")]
        (CpuStorage::U16(lhs), CpuStorage::U16(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::U32(lhs), CpuStorage::U32(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "u64")]
        (CpuStorage::U64(lhs), CpuStorage::U64(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::I8(lhs), CpuStorage::I8(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "i16")]
        (CpuStorage::I16(lhs), CpuStorage::I16(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        (CpuStorage::I32(lhs), CpuStorage::I32(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        #[cfg(feature = "i64")]
        (CpuStorage::I64(lhs), CpuStorage::I64(rhs), CpuStorage::BOOL(out)) => {
            call_kernel!(lhs, rhs, out)
        },
        _ => {
            return Err(HoduError::InternalError(
                "mismatched storage types in call_cmp".to_string(),
            ))
        },
    }

    Ok(output)
}
