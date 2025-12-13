//! Padding operations
//!
//! This module provides padding operations for tensors:
//! - pad_constant: Pad with a constant value
//! - pad_reflect: Pad with reflected values at boundaries
//! - pad_replicate: Pad by replicating edge values
//! - pad_circular: Pad with circular/wrapped values

use crate::{
    error::MetalKernelError,
    kernel::Kernels,
    kernels::macros::ops,
    metal::{Buffer, ComputeCommandEncoder, Device},
    set_params,
    source::Source,
    utils::{linear_split, BufferOffset, EncoderProvider},
};
use objc2_metal::MTLResourceUsage;

ops!(pad_constant, pad_reflect, pad_replicate, pad_circular);

/// Executes a constant padding operation.
///
/// Pads tensor with a constant value.
///
/// # Arguments
/// * `kernel` - Padding kernel (e.g., pad_constant::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer
/// * `output` - Output buffer (padded result)
/// * `pad_value` - Buffer containing the constant pad value
/// * `metadata` - Metadata describing tensor shapes and padding
///
/// # Metadata Layout
/// - `metadata[0]`: num_els (total number of output elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: input_shape
/// - `metadata[2+num_dims..2+2*num_dims]`: output_shape
/// - `metadata[2+2*num_dims..2+3*num_dims]`: pad_before (padding before each dim)
#[allow(clippy::too_many_arguments)]
pub fn call_ops_pad_constant(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    pad_value: BufferOffset,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Padding, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (&input, output, &pad_value, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.use_resource(pad_value.buffer, MTLResourceUsage::Read);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes a reflect padding operation.
///
/// Pads tensor with reflected values at boundaries.
/// For input [1, 2, 3] with pad=2: [3, 2, 1, 2, 3, 2, 1]
///
/// # Arguments
/// * `kernel` - Padding kernel (e.g., pad_reflect::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer
/// * `output` - Output buffer (padded result)
/// * `metadata` - Metadata describing tensor shapes and padding
///
/// # Metadata Layout
/// - `metadata[0]`: num_els (total number of output elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: input_shape
/// - `metadata[2+num_dims..2+2*num_dims]`: output_shape
/// - `metadata[2+2*num_dims..2+3*num_dims]`: pad_before (padding before each dim)
pub fn call_ops_pad_reflect(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Padding, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (&input, output, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes a replicate (edge) padding operation.
///
/// Pads tensor by replicating edge values.
/// For input [1, 2, 3] with pad=2: [1, 1, 1, 2, 3, 3, 3]
///
/// # Arguments
/// * `kernel` - Padding kernel (e.g., pad_replicate::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer
/// * `output` - Output buffer (padded result)
/// * `metadata` - Metadata describing tensor shapes and padding
///
/// # Metadata Layout
/// - `metadata[0]`: num_els (total number of output elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: input_shape
/// - `metadata[2+num_dims..2+2*num_dims]`: output_shape
/// - `metadata[2+2*num_dims..2+3*num_dims]`: pad_before (padding before each dim)
pub fn call_ops_pad_replicate(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Padding, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (&input, output, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes a circular (wrap) padding operation.
///
/// Pads tensor with circular/wrapped values.
/// For input [1, 2, 3] with pad=2: [2, 3, 1, 2, 3, 1, 2]
///
/// # Arguments
/// * `kernel` - Padding kernel (e.g., pad_circular::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer
/// * `output` - Output buffer (padded result)
/// * `metadata` - Metadata describing tensor shapes and padding
///
/// # Metadata Layout
/// - `metadata[0]`: num_els (total number of output elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: input_shape
/// - `metadata[2+num_dims..2+2*num_dims]`: output_shape
/// - `metadata[2+2*num_dims..2+3*num_dims]`: pad_before (padding before each dim)
pub fn call_ops_pad_circular(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Padding, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (&input, output, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
