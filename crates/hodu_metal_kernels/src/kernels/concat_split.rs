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

ops!(concat, split);

/// Call concat operation - concatenate multiple tensors along a dimension
#[allow(clippy::too_many_arguments)]
pub fn call_concat(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    output_shape: &[usize],
    concat_dim: usize,
    input_shapes: &[usize],
    input_strides: &[usize],
    input_offsets: &[usize],
    input_buffer_offsets: &[usize],
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::ConcatSplit, kernel_name.0)?;

    let num_dims = output_shape.len();
    let num_els: usize = output_shape.iter().product();
    let num_inputs = input_offsets.len();

    // Prepare metadata: output_shape, concat_dim, num_inputs, input_shapes, input_strides, input_offsets, input_buffer_offsets
    let mut metadata = Vec::with_capacity(
        num_dims + 2 + input_shapes.len() + input_strides.len() + input_offsets.len() + input_buffer_offsets.len(),
    );
    metadata.extend_from_slice(output_shape);
    metadata.push(concat_dim);
    metadata.push(num_inputs);
    metadata.extend_from_slice(input_shapes);
    metadata.extend_from_slice(input_strides);
    metadata.extend_from_slice(input_offsets);
    metadata.extend_from_slice(input_buffer_offsets);

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input (combined buffer with all inputs)
    // buffer(1): output
    // buffer(2): num_els
    // buffer(3): num_dims
    // buffer(4): metadata
    set_params!(encoder, (&input, output, num_els, num_dims, metadata.as_slice()));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Call split operation - split tensor into multiple outputs along a dimension
#[allow(clippy::too_many_arguments)]
pub fn call_split(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    input_shape: &[usize],
    input: BufferOffset,
    input_strides: &[usize],
    input_offset: usize,
    split_dim: usize,
    output_size_on_dim: usize,
    split_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::ConcatSplit, kernel_name.0)?;

    let num_dims = input_shape.len();

    // Calculate output shape
    let mut output_shape = input_shape.to_vec();
    output_shape[split_dim] = output_size_on_dim;
    let num_els: usize = output_shape.iter().product();

    // Prepare metadata: input_shape, input_strides, input_offset, split_dim, output_size_on_dim, split_offset
    let mut metadata = Vec::with_capacity(num_dims * 2 + 4);
    metadata.extend_from_slice(input_shape);
    metadata.extend_from_slice(input_strides);
    metadata.push(input_offset);
    metadata.push(split_dim);
    metadata.push(output_size_on_dim);
    metadata.push(split_offset);

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): output
    // buffer(2): num_els
    // buffer(3): num_dims
    // buffer(4): metadata
    set_params!(encoder, (&input, output, num_els, num_dims, metadata.as_slice()));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
