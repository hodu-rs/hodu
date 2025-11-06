use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::{MatrixOp, Op},
    types::{Layout, Shape},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_matmul(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<MetalStorage> {
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
    let m = lhs_shape.dims()[(lhs_ndim - 2) as usize];
    let k_lhs = lhs_shape.dims()[(lhs_ndim - 1) as usize];
    let k_rhs = rhs_shape.dims()[(rhs_ndim - 2) as usize];
    let n = rhs_shape.dims()[(rhs_ndim - 1) as usize];

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
    let mut batch_shape = Vec::with_capacity(batch_ndim as usize);
    for i in 0..batch_ndim {
        let lhs_idx = (lhs_batch_ndim as i32 - batch_ndim as i32 + i as i32) as usize;
        let rhs_idx = (rhs_batch_ndim as i32 - batch_ndim as i32 + i as i32) as usize;

        let lhs_dim = if lhs_idx < lhs_batch_ndim as usize {
            lhs_shape.dims()[lhs_idx]
        } else {
            1
        };
        let rhs_dim = if rhs_idx < rhs_batch_ndim as usize {
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

    // Calculate total number of output elements
    let num_els = output_shape.size();

    // Build metadata array for Metal kernel
    let mut metadata = Vec::with_capacity(
        4 + lhs_ndim as usize + rhs_ndim as usize + batch_ndim as usize + lhs_ndim as usize + rhs_ndim as usize + 5,
    );

    metadata.push(num_els as usize);
    metadata.push(lhs_ndim as usize);
    metadata.push(rhs_ndim as usize);
    metadata.push(batch_ndim as usize);

    // Add lhs shape
    for &dim in lhs_shape.dims() {
        metadata.push(dim as usize);
    }

    // Add rhs shape
    for &dim in rhs_shape.dims() {
        metadata.push(dim as usize);
    }

    // Add batch shape
    for &dim in &batch_shape {
        metadata.push(dim as usize);
    }

    // Add lhs strides
    for &stride in lhs_layout.strides() {
        metadata.push(stride as usize);
    }

    // Add rhs strides
    for &stride in rhs_layout.strides() {
        metadata.push(stride as usize);
    }

    // Add offsets
    metadata.push(lhs_layout.offset() as usize);
    metadata.push(rhs_layout.offset() as usize);

    // Add matrix dimensions M, K, N
    metadata.push(m as usize);
    metadata.push(k_lhs as usize);
    metadata.push(n as usize);

    let dtype = lhs_storage.dtype();
    let device = lhs_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els as usize, dtype, "matmul_output")?;

    // Generate kernel name
    let kernel_name = format!("matmul_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets for inputs
    let lhs_offset = BufferOffset::zero_offset(lhs_storage.buffer());
    let rhs_offset = BufferOffset::zero_offset(rhs_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_matmul(
        device.device(),
        &command_buffer,
        device.kernels(),
        kernel,
        lhs_offset,
        rhs_offset,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(
        output_buffer,
        device.clone(),
        num_els as usize,
        dtype,
    ))
}

pub fn call_dot(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<MetalStorage> {
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
    let num_els = output_shape.size();

    // Build metadata array for Metal kernel
    let mut metadata = Vec::with_capacity(9);

    metadata.push(m as usize);
    metadata.push(k_lhs as usize);
    metadata.push(n as usize);

    // Add strides
    let lhs_strides = lhs_layout.strides();
    let rhs_strides = rhs_layout.strides();

    metadata.push(lhs_strides[0] as usize); // lhs_stride_m
    metadata.push(lhs_strides[1] as usize); // lhs_stride_k
    metadata.push(rhs_strides[0] as usize); // rhs_stride_k
    metadata.push(rhs_strides[1] as usize); // rhs_stride_n

    // Add offsets
    metadata.push(lhs_layout.offset() as usize);
    metadata.push(rhs_layout.offset() as usize);

    let dtype = lhs_storage.dtype();
    let device = lhs_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els as usize, dtype, "dot_output")?;

    // Generate kernel name
    let kernel_name = format!("dot_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets for inputs
    let lhs_offset = BufferOffset::zero_offset(lhs_storage.buffer());
    let rhs_offset = BufferOffset::zero_offset(rhs_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_dot(
        device.device(),
        &command_buffer,
        device.kernels(),
        kernel,
        lhs_offset,
        rhs_offset,
        &output_buffer,
        m as usize,
        n as usize,
        &metadata,
    )?;

    Ok(MetalStorage::new(
        output_buffer,
        device.clone(),
        num_els as usize,
        dtype,
    ))
}
