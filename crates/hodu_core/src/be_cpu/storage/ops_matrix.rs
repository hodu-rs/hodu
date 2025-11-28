use crate::{
    be::{device::BackendDeviceT, storage::BackendStorageT},
    be_cpu::{device::CpuDevice, storage::CpuStorage},
    compat::*,
    error::{HoduError, HoduResult},
    ops::{MatrixOp, Op},
    types::{Layout, Shape},
};
use core::ffi::c_void;

/// Execute matmul operation with batched matrix multiplication
///
/// Performs C = A @ B with broadcasting support for batch dimensions.
/// Handles arbitrary batch dimensions and automatically broadcasts dimensions of size 1.
///
/// # Arguments
/// * `lhs_storage` - Left-hand side tensor storage (A)
/// * `rhs_storage` - Right-hand side tensor storage (B)
/// * `lhs_layout` - Layout of left-hand side tensor
/// * `rhs_layout` - Layout of right-hand side tensor
/// * `op` - The matrix operation (should be Op::Matrix(MatrixOp::Matmul))
///
/// # Returns
/// Output storage containing the result of the matrix multiplication
pub fn call_ops_matmul(
    lhs_storage: &CpuStorage,
    rhs_storage: &CpuStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Validate op
    match op {
        Op::Matrix(MatrixOp::Matmul) => (),
        _ => return Err(HoduError::BackendError("Lcall_matmulE expects LMatmulE op".to_string())),
    };
    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let lhs_ndim = lhs_shape.ndim();
    let rhs_ndim = rhs_shape.ndim();

    // Validate shapes for matmul
    if lhs_ndim < 2 || rhs_ndim < 2 {
        return Err(HoduError::BackendError(
            "matmul requires at least 2D tensors".to_string(),
        ));
    }

    // Extract matrix dimensions
    let m = lhs_shape.dims()[lhs_ndim - 2];
    let k_lhs = lhs_shape.dims()[lhs_ndim - 1];
    let k_rhs = rhs_shape.dims()[rhs_ndim - 2];
    let n = rhs_shape.dims()[rhs_ndim - 1];

    // Check that inner dimensions match
    if k_lhs != k_rhs {
        return Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.clone(),
            rhs: rhs_shape.clone(),
            op: crate::ops::Op::Matrix(crate::ops::MatrixOp::Matmul),
        });
    }

    // Compute batch dimensions and output shape
    let lhs_batch_ndim = lhs_ndim - 2;
    let rhs_batch_ndim = rhs_ndim - 2;
    let batch_ndim = lhs_batch_ndim.max(rhs_batch_ndim);

    // Broadcast batch dimensions
    let mut batch_shape = Vec::with_capacity(batch_ndim);
    for i in 0..batch_ndim {
        let lhs_idx = (lhs_batch_ndim as i32 - batch_ndim as i32 + i as i32) as usize;
        let rhs_idx = (rhs_batch_ndim as i32 - batch_ndim as i32 + i as i32) as usize;

        let lhs_dim = if lhs_idx < lhs_batch_ndim {
            lhs_shape.dims()[lhs_idx]
        } else {
            1
        };
        let rhs_dim = if rhs_idx < rhs_batch_ndim {
            rhs_shape.dims()[rhs_idx]
        } else {
            1
        };

        if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
            return Err(HoduError::IncompatibleShapes {
                lhs: lhs_shape.clone(),
                rhs: rhs_shape.clone(),
                op: crate::ops::Op::Matrix(crate::ops::MatrixOp::Matmul),
            });
        }

        batch_shape.push(lhs_dim.max(rhs_dim));
    }

    // Build output shape: [...batch_dims, M, N]
    let mut output_shape_vec = batch_shape.clone();
    output_shape_vec.push(m);
    output_shape_vec.push(n);
    let output_shape = Shape::new(&output_shape_vec);

    // Create output layout for metadata generation
    let output_layout = Layout::from_shape(&output_shape);

    // Generate metadata using centralized function
    let metadata = crate::op_metadatas::matmul_metadata(lhs_layout, rhs_layout, &output_layout)?;

    // Generate kernel name
    let kernel_name = format!("hodu_cpu_matmul_{}", lhs_storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let dtype = lhs_storage.dtype();
    let mut output = CpuDevice::allocate(output_shape.size(), dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($lhs_data:expr, $rhs_data:expr, $out_data:expr) => {{
            let lhs_ptr = $lhs_data.as_ptr() as *const c_void;
            let rhs_ptr = $rhs_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_matmul(kernel, lhs_ptr, rhs_ptr, out_ptr, &metadata)?;
        }};
    }

    match (lhs_storage, rhs_storage, &mut output) {
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
            return Err(HoduError::BackendError(
                "mismatched storage types in call_matmul".to_string(),
            ))
        },
    }

    Ok(output)
}

/// Execute dot operation for 2D matrix multiplication
///
/// Performs optimized C = A @ B for two 2D matrices only.
/// This is a simplified version without batch dimensions.
///
/// # Arguments
/// * `lhs_storage` - Left-hand side matrix storage (A)
/// * `rhs_storage` - Right-hand side matrix storage (B)
/// * `lhs_layout` - Layout of left-hand side matrix
/// * `rhs_layout` - Layout of right-hand side matrix
/// * `op` - The matrix operation (should be Op::Matrix(MatrixOp::Dot))
///
/// # Returns
/// Output storage containing the result of the matrix multiplication
pub fn call_ops_dot(
    lhs_storage: &CpuStorage,
    rhs_storage: &CpuStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<CpuStorage> {
    // Validate op
    match op {
        Op::Matrix(MatrixOp::Dot) => (),
        _ => return Err(HoduError::BackendError("Lcall_dotE expects LDotE op".to_string())),
    };
    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let lhs_ndim = lhs_shape.ndim();
    let rhs_ndim = rhs_shape.ndim();

    // Validate that both are 2D matrices
    if lhs_ndim != 2 || rhs_ndim != 2 {
        return Err(HoduError::BackendError("dot requires exactly 2D tensors".to_string()));
    }

    // Extract matrix dimensions
    let m = lhs_shape.dims()[0];
    let k_lhs = lhs_shape.dims()[1];
    let k_rhs = rhs_shape.dims()[0];
    let n = rhs_shape.dims()[1];

    // Check that inner dimensions match
    if k_lhs != k_rhs {
        return Err(HoduError::IncompatibleShapes {
            lhs: lhs_shape.clone(),
            rhs: rhs_shape.clone(),
            op: crate::ops::Op::Matrix(crate::ops::MatrixOp::Dot),
        });
    }

    // Build output shape [M, N]
    let output_shape = Shape::new(&[m, n]);

    // Generate metadata using centralized function
    let metadata = crate::op_metadatas::dot_metadata(lhs_layout, rhs_layout)?;

    // Generate kernel name
    let kernel_name = format!("hodu_cpu_dot_{}", lhs_storage.dtype());
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = hodu_cpu_kernels::macros::Kernel(kernel_name_static);

    // Create output storage
    let dtype = lhs_storage.dtype();
    let mut output = CpuDevice::allocate(output_shape.size(), dtype)?;

    // Get raw pointers and call kernel
    macro_rules! call_kernel {
        ($lhs_data:expr, $rhs_data:expr, $out_data:expr) => {{
            let lhs_ptr = $lhs_data.as_ptr() as *const c_void;
            let rhs_ptr = $rhs_data.as_ptr() as *const c_void;
            let out_ptr = $out_data.as_mut_ptr() as *mut c_void;

            hodu_cpu_kernels::call_ops_dot(kernel, lhs_ptr, rhs_ptr, out_ptr, &metadata)?;
        }};
    }

    match (lhs_storage, rhs_storage, &mut output) {
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
            return Err(HoduError::BackendError(
                "mismatched storage types in call_dot".to_string(),
            ))
        },
    }

    Ok(output)
}
