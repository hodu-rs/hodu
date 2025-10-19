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
    reduce_window_max,
    reduce_window_min,
    reduce_window_sum,
    reduce_window_mean
);

#[allow(clippy::too_many_arguments)]
pub fn call_reduce_window(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    input_shape: &[usize],
    input: BufferOffset,
    input_strides: &[usize],
    input_offset: usize,
    window_shape: &[usize],
    strides: &[usize],
    padding: &[usize],
    output_shape: &[usize],
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Windowing, kernel_name.0)?;

    let num_dims = input_shape.len();
    let num_els: usize = output_shape.iter().product();

    // Prepare metadata: input_shape, input_strides, offset, window_shape, strides, padding, output_shape
    // Metadata layout (in buffer(4)):
    // - input_shape: [num_dims] dimensions of input tensor
    // - input_strides: [num_dims] strides of input tensor
    // - offset: single value, offset into input buffer
    // - window_shape: [num_dims] size of reduction window
    // - strides: [num_dims] stride of window movement
    // - padding: [num_dims * 2] padding (before, after) for each dimension
    // - output_shape: [num_dims] dimensions of output tensor
    let mut metadata = Vec::with_capacity(num_dims * 7 + 1);
    metadata.extend_from_slice(input_shape);
    metadata.extend_from_slice(input_strides);
    metadata.push(input_offset);
    metadata.extend_from_slice(window_shape);
    metadata.extend_from_slice(strides);
    metadata.extend_from_slice(padding);
    metadata.extend_from_slice(output_shape);

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): output
    // buffer(2): num_els (output elements)
    // buffer(3): num_dims
    // buffer(4): metadata
    set_params!(encoder, (&input, output, num_els, num_dims, metadata.as_slice()));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
