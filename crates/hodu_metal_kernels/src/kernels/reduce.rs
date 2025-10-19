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

ops!(
    reduce_sum,
    reduce_max,
    reduce_min,
    reduce_prod,
    reduce_mean,
    reduce_norm,
    reduce_argmax,
    reduce_argmin,
    reduce_any,
    reduce_all
);

#[allow(clippy::too_many_arguments)]
pub fn call_reduce(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    shape: &[usize],
    input: BufferOffset,
    input_strides: &[usize],
    input_offset: usize,
    output_shape: &[usize],
    reduce_dims: &[usize],
    reduce_size: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name.0)?;

    let num_dims = shape.len();
    let num_els: usize = output_shape.iter().product();
    let num_reduce_dims = reduce_dims.len();

    // Prepare metadata: dims, strides, offset, output_shape, reduce_dims, num_reduce_dims
    let mut metadata = Vec::with_capacity(num_dims * 4 + 2 + num_reduce_dims);
    metadata.extend_from_slice(shape);
    metadata.extend_from_slice(input_strides);
    metadata.push(input_offset);
    metadata.extend_from_slice(output_shape);
    metadata.extend_from_slice(reduce_dims);
    metadata.push(num_reduce_dims);

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): output
    // buffer(2): num_els (output elements)
    // buffer(3): num_dims
    // buffer(4): metadata (dims, strides, offset, output_shape, reduce_dims, num_reduce_dims)
    // buffer(5): reduce_size
    set_params!(
        encoder,
        (&input, output, num_els, num_dims, metadata.as_slice(), reduce_size)
    );

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
