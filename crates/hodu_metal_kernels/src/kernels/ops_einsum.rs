//! Einsum operations
//!
//! This module provides Einstein summation operations for tensor contractions.

use crate::{
    error::MetalKernelError,
    kernel::Kernels,
    kernels::macros::ops,
    metal::{Buffer, ComputeCommandEncoder, Device},
    source::Source,
    utils::{linear_split, BufferOffset, EncoderProvider},
};
use objc2_metal::MTLResourceUsage;

ops!(einsum);

/// Executes an einsum operation.
///
/// # Arguments
/// * `kernel` - Einsum kernel (e.g., einsum::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `inputs` - Input tensor buffers (up to 4)
/// * `output` - Output buffer
/// * `metadata` - Metadata describing tensor shapes and einsum specification
pub fn call_ops_einsum(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    inputs: &[BufferOffset],
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Einsum, kernel.0)?;

    let num_output_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Set input buffers (up to 4)
    for (i, input) in inputs.iter().enumerate() {
        encoder.set_buffer(i, Some(input.buffer), input.offset_in_bytes);
        encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    }
    // Set remaining input buffers to the first input (unused but required by shader)
    for i in inputs.len()..4 {
        encoder.set_buffer(i, Some(inputs[0].buffer), inputs[0].offset_in_bytes);
    }

    // Set output buffer at index 4
    encoder.set_buffer(4, Some(output), 0);
    encoder.use_resource(output, MTLResourceUsage::Write);

    // Set metadata at index 5
    crate::utils::set_param(encoder, 5, metadata);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_output_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
