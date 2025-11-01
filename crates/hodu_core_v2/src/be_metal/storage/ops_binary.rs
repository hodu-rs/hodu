use crate::{
    be::storage::BackendStorageT,
    be_metal::storage::MetalStorage,
    error::{HoduError, HoduResult},
    ops::Op,
    types::{DType, Layout},
};
use hodu_metal_kernels::{kernels, utils::BufferOffset};

pub fn call_binary(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Extract binary op
    let binary_op = match op {
        Op::Binary(binary_op) => binary_op,
        _ => return Err(HoduError::InternalError("call_binary expects binary op".to_string())),
    };

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let num_els = lhs_shape.size();
    let num_dims = lhs_shape.ndim();

    // Build metadata array for Metal kernel
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

    let dtype = lhs_storage.dtype();
    let device = lhs_storage.backend_device();

    // Create output buffer
    let output_buffer = device.new_buffer(num_els as usize, dtype, "binary_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", binary_op, dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets for inputs
    let lhs_offset = BufferOffset::zero_offset(lhs_storage.buffer());
    let rhs_offset = BufferOffset::zero_offset(rhs_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_binary(
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

pub fn call_binary_logical(
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
            return Err(HoduError::InternalError(
                "call_binary_logical expects binary logical op".to_string(),
            ))
        },
    };

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let num_els = lhs_shape.size();
    let num_dims = lhs_shape.ndim();

    // Build metadata array for Metal kernel
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

    let input_dtype = lhs_storage.dtype();
    let device = lhs_storage.backend_device();

    // Create output buffer (logical ops return BOOL)
    let output_dtype = DType::BOOL;
    let output_buffer = device.new_buffer(num_els as usize, output_dtype, "binary_logical_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", binary_op, input_dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets for inputs
    let lhs_offset = BufferOffset::zero_offset(lhs_storage.buffer());
    let rhs_offset = BufferOffset::zero_offset(rhs_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_binary(
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
        output_dtype,
    ))
}

pub fn call_cmp(
    lhs_storage: &MetalStorage,
    rhs_storage: &MetalStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> HoduResult<MetalStorage> {
    // Extract cmp op
    let cmp_op = match op {
        Op::Cmp(cmp_op) => cmp_op,
        _ => return Err(HoduError::InternalError("call_cmp expects cmp op".to_string())),
    };

    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let num_els = lhs_shape.size();
    let num_dims = lhs_shape.ndim();

    // Build metadata array for Metal kernel
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

    let input_dtype = lhs_storage.dtype();
    let device = lhs_storage.backend_device();

    // Create output buffer (cmp ops return BOOL)
    let output_dtype = DType::BOOL;
    let output_buffer = device.new_buffer(num_els as usize, output_dtype, "cmp_output")?;

    // Get kernel name
    let kernel_name = format!("{}_{}", cmp_op, input_dtype);
    let kernel_name_static = crate::cache::kernel::get_kernel_name(kernel_name);
    let kernel = kernels::Kernel(kernel_name_static);

    // Create buffer offsets for inputs
    let lhs_offset = BufferOffset::zero_offset(lhs_storage.buffer());
    let rhs_offset = BufferOffset::zero_offset(rhs_storage.buffer());

    // Get command buffer and call kernel
    let command_buffer = device.command_buffer()?;
    kernels::call_binary(
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
        output_dtype,
    ))
}
