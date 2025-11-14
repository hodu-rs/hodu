use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    types::{DType, Layout},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_ops_binary(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Extract binary op
    let binary_op = match op {
        Op::Binary(binary_op) => binary_op,
        _ => return Err(HoduError::BackendError("call_ops_binary expects binary op".to_string())),
    };

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let num_els = lhs_shape.size();
    let num_dims = lhs_shape.ndim();

    // Build metadata array for Metal kernel
    let mut metadata = Vec::with_capacity(2 + 4 * num_dims + 2);
    metadata.push(num_els);
    metadata.push(num_dims);

    // Add shapes
    for &dim in lhs_shape.dims() {
        metadata.push(dim);
    }
    for &dim in rhs_shape.dims() {
        metadata.push(dim);
    }

    // Add strides
    for &stride in lhs_layout.strides() {
        metadata.push(stride);
    }
    for &stride in rhs_layout.strides() {
        metadata.push(stride);
    }

    // Add offsets
    metadata.push(lhs_layout.offset());
    metadata.push(rhs_layout.offset());

    let dtype = lhs_storage.dtype();
    let device = lhs_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els, dtype, "binary_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", binary_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets for inputs
    let lhs_offset = BufferOffset::zero_offset(lhs_storage.buffer());
    let rhs_offset = BufferOffset::zero_offset(rhs_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_binary(
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

pub fn call_ops_binary_logical(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Extract binary logical op
    let binary_op = match op {
        Op::BinaryLogical(binary_op) => binary_op,
        _ => {
            return Err(HoduError::BackendError(
                "call_ops_binary_logical expects binary logical op".to_string(),
            ))
        },
    };

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let num_els = lhs_shape.size();
    let num_dims = lhs_shape.ndim();

    // Build metadata array for Metal kernel
    let mut metadata = Vec::with_capacity(2 + 4 * num_dims + 2);
    metadata.push(num_els);
    metadata.push(num_dims);

    // Add shapes
    for &dim in lhs_shape.dims() {
        metadata.push(dim);
    }
    for &dim in rhs_shape.dims() {
        metadata.push(dim);
    }

    // Add strides
    for &stride in lhs_layout.strides() {
        metadata.push(stride);
    }
    for &stride in rhs_layout.strides() {
        metadata.push(stride);
    }

    // Add offsets
    metadata.push(lhs_layout.offset());
    metadata.push(rhs_layout.offset());

    let input_dtype = lhs_storage.dtype();
    let device = lhs_storage.backend_device();

    // Create output buffer (logical ops return BOOL)
    let output_dtype = DType::BOOL;
    let output_buffer = device.new_buffer(num_els, output_dtype, "binary_logical_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", binary_op, input_dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets for inputs
    let lhs_offset = BufferOffset::zero_offset(lhs_storage.buffer());
    let rhs_offset = BufferOffset::zero_offset(rhs_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_binary(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        lhs_offset,
        rhs_offset,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, output_dtype))
}

pub fn call_ops_cmp(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Extract cmp op
    let cmp_op = match op {
        Op::Cmp(cmp_op) => cmp_op,
        _ => return Err(HoduError::BackendError("Lcall_ops_cmpE expects LcmpE op".to_string())),
    };

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let num_els = lhs_shape.size();
    let num_dims = lhs_shape.ndim();

    // Build metadata array for Metal kernel
    let mut metadata = Vec::with_capacity(2 + 4 * num_dims + 2);
    metadata.push(num_els);
    metadata.push(num_dims);

    // Add shapes
    for &dim in lhs_shape.dims() {
        metadata.push(dim);
    }
    for &dim in rhs_shape.dims() {
        metadata.push(dim);
    }

    // Add strides
    for &stride in lhs_layout.strides() {
        metadata.push(stride);
    }
    for &stride in rhs_layout.strides() {
        metadata.push(stride);
    }

    // Add offsets
    metadata.push(lhs_layout.offset());
    metadata.push(rhs_layout.offset());

    let input_dtype = lhs_storage.dtype();
    let device = lhs_storage.backend_device();

    // Create output buffer (cmp ops return BOOL)
    let output_dtype = DType::BOOL;
    let output_buffer = device.new_buffer(num_els, output_dtype, "cmp_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", cmp_op, input_dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets for inputs
    let lhs_offset = BufferOffset::zero_offset(lhs_storage.buffer());
    let rhs_offset = BufferOffset::zero_offset(rhs_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_ops_binary(
        kernel,
        device.kernels(),
        device.device(),
        &command_buffer,
        lhs_offset,
        rhs_offset,
        &output_buffer,
        &metadata,
    )?;

    Ok(MetalStorage::new(output_buffer, device.clone(), num_els, output_dtype))
}
