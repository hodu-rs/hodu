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
    index_select,
    index_put,
    gather,
    scatter,
    scatter_add,
    scatter_max,
    scatter_min
);

/// Call index_select operation
#[allow(clippy::too_many_arguments)]
pub fn call_index_select(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    shape: &[usize],
    input: BufferOffset,
    input_strides: &[usize],
    input_offset: usize,
    indices: BufferOffset,
    dim: usize,
    num_indices: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel_name.0)?;

    let num_dims = shape.len();

    // Calculate output shape (same as input but with dim replaced by num_indices)
    let mut output_shape = shape.to_vec();
    output_shape[dim] = num_indices;
    let num_els: usize = output_shape.iter().product();

    // Prepare metadata: input_shape, input_strides, input_offset, dim, num_indices
    let mut metadata = Vec::with_capacity(num_dims * 2 + 3);
    metadata.extend_from_slice(shape);
    metadata.extend_from_slice(input_strides);
    metadata.push(input_offset);
    metadata.push(dim);
    metadata.push(num_indices);

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): indices
    // buffer(2): output
    // buffer(3): num_els
    // buffer(4): num_dims
    // buffer(5): metadata
    set_params!(
        encoder,
        (&input, &indices, output, num_els, num_dims, metadata.as_slice())
    );

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(indices.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Call index_put operation
#[allow(clippy::too_many_arguments)]
pub fn call_index_put(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    input_shape: &[usize],
    input: BufferOffset,
    input_strides: &[usize],
    input_offset: usize,
    indices: BufferOffset,
    values: BufferOffset,
    values_strides: &[usize],
    values_offset: usize,
    dim: usize,
    num_indices: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel_name.0)?;

    let num_dims = input_shape.len();
    let num_els: usize = input_shape.iter().product();

    // Prepare metadata: input_shape, input_strides, values_strides, input_offset, values_offset, dim, num_indices
    let mut metadata = Vec::with_capacity(num_dims * 3 + 4);
    metadata.extend_from_slice(input_shape);
    metadata.extend_from_slice(input_strides);
    metadata.extend_from_slice(values_strides);
    metadata.push(input_offset);
    metadata.push(values_offset);
    metadata.push(dim);
    metadata.push(num_indices);

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): indices
    // buffer(2): values
    // buffer(3): output
    // buffer(4): num_els
    // buffer(5): num_dims
    // buffer(6): metadata
    set_params!(
        encoder,
        (
            &input,
            &indices,
            &values,
            output,
            num_els,
            num_dims,
            metadata.as_slice()
        )
    );

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(indices.buffer, MTLResourceUsage::Read);
    encoder.use_resource(values.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Call gather operation
#[allow(clippy::too_many_arguments)]
pub fn call_gather(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    input_shape: &[usize],
    input: BufferOffset,
    input_strides: &[usize],
    input_offset: usize,
    indices: BufferOffset,
    indices_strides: &[usize],
    indices_offset: usize,
    dim: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel_name.0)?;

    let num_dims = input_shape.len();
    let num_els: usize = input_shape.iter().product(); // Output has same shape as indices

    // Prepare metadata: input_shape, input_strides, indices_strides, input_offset, indices_offset, dim
    let mut metadata = Vec::with_capacity(num_dims * 3 + 3);
    metadata.extend_from_slice(input_shape);
    metadata.extend_from_slice(input_strides);
    metadata.extend_from_slice(indices_strides);
    metadata.push(input_offset);
    metadata.push(indices_offset);
    metadata.push(dim);

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): indices
    // buffer(2): output
    // buffer(3): num_els
    // buffer(4): num_dims
    // buffer(5): metadata
    set_params!(
        encoder,
        (&input, &indices, output, num_els, num_dims, metadata.as_slice())
    );

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(indices.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Call scatter operation (and variants: scatter, scatter_add, scatter_max, scatter_min)
#[allow(clippy::too_many_arguments)]
pub fn call_scatter(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    input_shape: &[usize],
    input: BufferOffset,
    input_strides: &[usize],
    input_offset: usize,
    indices: BufferOffset,
    indices_strides: &[usize],
    indices_offset: usize,
    src: BufferOffset,
    src_shape: &[usize],
    src_strides: &[usize],
    src_offset: usize,
    dim: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel_name.0)?;

    let num_dims = input_shape.len();
    let num_els: usize = src_shape.iter().product();

    // Prepare metadata: input_shape, input_strides, src_shape, src_strides, indices_strides,
    //                   input_offset, src_offset, indices_offset, dim
    let mut metadata = Vec::with_capacity(num_dims * 5 + 4);
    metadata.extend_from_slice(input_shape);
    metadata.extend_from_slice(input_strides);
    metadata.extend_from_slice(src_shape);
    metadata.extend_from_slice(src_strides);
    metadata.extend_from_slice(indices_strides);
    metadata.push(input_offset);
    metadata.push(src_offset);
    metadata.push(indices_offset);
    metadata.push(dim);

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): indices
    // buffer(2): src
    // buffer(3): output
    // buffer(4): num_els
    // buffer(5): num_dims
    // buffer(6): metadata
    set_params!(
        encoder,
        (&input, &indices, &src, output, num_els, num_dims, metadata.as_slice())
    );

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(indices.buffer, MTLResourceUsage::Read);
    encoder.use_resource(src.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
