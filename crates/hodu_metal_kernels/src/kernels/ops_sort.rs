//! Sort operations
//!
//! This module provides sorting operations:
//! - topk: Get top-k largest or smallest elements along a dimension

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

ops!(topk);

/// Executes a topk operation.
///
/// Returns the k largest or smallest elements along the last dimension.
///
/// # Arguments
/// * `kernel` - TopK kernel (e.g., topk::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer
/// * `values` - Output buffer for values
/// * `indices` - Output buffer for indices (i32)
/// * `metadata` - Metadata describing the operation
///
/// # Metadata Layout
/// - `metadata[0]`: output_size (k * outer_size)
/// - `metadata[1]`: k (number of top elements)
/// - `metadata[2]`: last_dim_size (size of the dimension to search along)
/// - `metadata[3]`: outer_size (product of all dimensions except last)
/// - `metadata[4]`: largest (1 = largest, 0 = smallest)
/// - `metadata[5]`: sorted (1 = sorted, 0 = unsorted)
/// - `metadata[6]`: offset
#[allow(clippy::too_many_arguments)]
pub fn call_topk(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    values: &Buffer,
    indices: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Sort, kernel.0)?;

    let outer_size = metadata[3];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (&input, values, indices, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(values, MTLResourceUsage::Write);
    encoder.use_resource(indices, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, outer_size);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
