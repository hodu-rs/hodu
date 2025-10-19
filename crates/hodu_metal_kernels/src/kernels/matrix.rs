use crate::{
    error::MetalKernelError,
    kernel::Kernels,
    kernels::macros::ops,
    metal::{Buffer, ComputeCommandEncoder, Device},
    set_params,
    source::Source,
    utils::{linear_split, BufferOffset, EncoderProvider},
};
use objc2_metal::{MTLResourceUsage, MTLSize};

ops!(matmul, dot);

/// Call matmul operation - batched matrix multiplication with broadcasting
#[allow(clippy::too_many_arguments)]
pub fn call_matmul(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    lhs: BufferOffset,
    rhs: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Matrix, kernel_name.0)?;

    let num_els = metadata[0]; // First element should be num_els

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): lhs
    // buffer(1): rhs
    // buffer(2): output
    // buffer(3): num_els
    // buffer(4): metadata (lhs_ndim, rhs_ndim, batch_ndim, shapes, strides, offsets, M, K, N)
    set_params!(encoder, (&lhs, &rhs, output, num_els, &metadata[1..]));

    encoder.use_resource(lhs.buffer, MTLResourceUsage::Read);
    encoder.use_resource(rhs.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Call dot operation - tiled 2D matrix multiplication (optimized with threadgroup memory)
#[allow(clippy::too_many_arguments)]
pub fn call_dot(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    lhs: BufferOffset,
    rhs: BufferOffset,
    output: &Buffer,
    m: usize,
    n: usize,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Matrix, kernel_name.0)?;

    let num_els = m * n;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): lhs
    // buffer(1): rhs
    // buffer(2): output
    // buffer(3): num_els
    // buffer(4): metadata (M, K, N, lhs_strides, rhs_strides, lhs_offset, rhs_offset)
    set_params!(encoder, (&lhs, &rhs, output, num_els, metadata));

    encoder.use_resource(lhs.buffer, MTLResourceUsage::Read);
    encoder.use_resource(rhs.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    // For tiled dot product, use 2D thread groups (16x16 tiles)
    const TILE_SIZE: usize = 16;
    let threadgroup_size = MTLSize {
        width: TILE_SIZE,
        height: TILE_SIZE,
        depth: 1,
    };

    let grid_width = (n + TILE_SIZE - 1) / TILE_SIZE;
    let grid_height = (m + TILE_SIZE - 1) / TILE_SIZE;

    let threadgroup_count = MTLSize {
        width: grid_width,
        height: grid_height,
        depth: 1,
    };

    encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);

    Ok(())
}
