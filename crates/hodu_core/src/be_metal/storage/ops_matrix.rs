use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::{MatrixOp, Op},
    types::{Layout, Shape},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_ops_matmul(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    match op {
        Op::Matrix(MatrixOp::Matmul) => (),
        _ => {
            return Err(HoduError::BackendError(
                "Lcall_ops_matmulE expects LMatmulE op".to_string(),
            ))
        },
    };

    // Compute output shape for matmul
    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let lhs_ndim = lhs_shape.ndim();
    let rhs_ndim = rhs_shape.ndim();

    if lhs_ndim < 2 || rhs_ndim < 2 {
        return Err(HoduError::BackendError(
            "matmul requires at least 2D tensors".to_string(),
        ));
    }

    let m = lhs_shape.dims()[lhs_ndim - 2];
    let n = rhs_shape.dims()[rhs_ndim - 1];

    let lhs_batch_ndim = lhs_ndim - 2;
    let rhs_batch_ndim = rhs_ndim - 2;
    let batch_ndim = lhs_batch_ndim.max(rhs_batch_ndim);

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

        batch_shape.push(lhs_dim.max(rhs_dim));
    }

    let mut output_shape_vec = batch_shape;
    output_shape_vec.push(m);
    output_shape_vec.push(n);
    let output_shape = Shape::new(&output_shape_vec);
    let output_layout = Layout::from_shape(&output_shape);

    let metadata = crate::op_metadatas::matmul_metadata(lhs_layout, rhs_layout, &output_layout)?;
    let num_els = output_shape.size();

    let dtype = lhs_storage.dtype();
    let device = lhs_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els, dtype, "matmul_output")?;

    // Generate kernel name
    let kernel_name = format!("matmul_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets for inputs
    let lhs_offset = BufferOffset::zero_offset(lhs_storage.buffer());
    let rhs_offset = BufferOffset::zero_offset(rhs_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_matmul(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        lhs_offset,
        rhs_offset,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, dtype))
}

pub fn call_ops_dot(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Validate op
    match op {
        Op::Matrix(MatrixOp::Dot) => (),
        _ => return Err(HoduError::BackendError("Lcall_ops_dotE expects LDotE op".to_string())),
    };

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();

    if lhs_shape.ndim() != 2 || rhs_shape.ndim() != 2 {
        return Err(HoduError::BackendError("dot requires exactly 2D tensors".to_string()));
    }

    let m = lhs_shape.dims()[0];
    let n = rhs_shape.dims()[1];
    let output_shape = Shape::new(&[m, n]);

    let metadata = crate::op_metadatas::dot_metadata(lhs_layout, rhs_layout)?;
    let num_els = output_shape.size();

    let dtype = lhs_storage.dtype();
    let device = lhs_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els, dtype, "dot_output")?;

    // Generate kernel name
    let kernel_name = format!("dot_{}", dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets for inputs
    let lhs_offset = BufferOffset::zero_offset(lhs_storage.buffer());
    let rhs_offset = BufferOffset::zero_offset(rhs_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_dot(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        lhs_offset,
        rhs_offset,
        &output_buffer,
        m,
        n,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, dtype))
}
