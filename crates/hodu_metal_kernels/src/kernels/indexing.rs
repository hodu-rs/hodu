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

/// Executes an index_select operation to extract elements along a dimension using indices.
///
/// # Arguments
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `kernels` - Kernel cache
/// * `kernel_name` - Index select kernel (e.g., index_select::F32)
/// * `shape` - Shape of input tensor
/// * `input` - Input tensor buffer
/// * `input_strides` - Strides of input tensor
/// * `input_offset` - Starting offset in input buffer
/// * `indices` - Index tensor buffer (contains indices to select)
/// * `dim` - Dimension along which to select
/// * `num_indices` - Number of indices (length of indices tensor)
/// * `output` - Output buffer
///
/// # Metadata Layout
/// Total metadata length: `2 + num_dims * 2 + 3`
///
/// - `metadata[0]`: num_els (total output elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: input_shape
/// - `metadata[2+num_dims..2+2*num_dims]`: input_strides
/// - `metadata[2+2*num_dims]`: input_offset
/// - `metadata[2+2*num_dims+1]`: dim (dimension to select along)
/// - `metadata[2+2*num_dims+2]`: num_indices
///
/// # Example
/// ```ignore
/// // Select elements at indices [0, 2] from dimension 0 of a 4x3 tensor
/// call_index_select(
///     &device, &command_buffer, &kernels, index_select::F32,
///     &[4, 3], input_buffer, &[3, 1], 0, indices_buffer, 0, 2, &output
/// )?;
/// // Output shape will be [2, 3]
/// ```
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
    let mut metadata = Vec::with_capacity(2 + num_dims * 2 + 3);
    metadata.push(num_els);
    metadata.push(num_dims);
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
    // buffer(3): metadata
    set_params!(encoder, (&input, &indices, output, metadata.as_slice()));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(indices.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes an index_put operation to write values into a tensor at specified indices along a dimension.
///
/// # Arguments
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `kernels` - Kernel cache
/// * `kernel_name` - Index put kernel (e.g., index_put::F32)
/// * `input_shape` - Shape of input tensor
/// * `input` - Input tensor buffer
/// * `input_strides` - Strides of input tensor
/// * `input_offset` - Starting offset in input buffer
/// * `indices` - Index tensor buffer (specifies where to put values)
/// * `values` - Values tensor buffer (values to write)
/// * `values_strides` - Strides of values tensor
/// * `values_offset` - Starting offset in values buffer
/// * `dim` - Dimension along which to put values
/// * `num_indices` - Number of indices
/// * `output` - Output buffer (modified copy of input)
///
/// # Metadata Layout
/// Total metadata length: `2 + num_dims * 3 + 4`
///
/// - `metadata[0]`: num_els (total elements in output)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: input_shape
/// - `metadata[2+num_dims..2+2*num_dims]`: input_strides
/// - `metadata[2+2*num_dims..2+3*num_dims]`: values_strides
/// - `metadata[2+3*num_dims]`: input_offset
/// - `metadata[2+3*num_dims+1]`: values_offset
/// - `metadata[2+3*num_dims+2]`: dim (dimension to put along)
/// - `metadata[2+3*num_dims+3]`: num_indices
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
    let mut metadata = Vec::with_capacity(2 + num_dims * 3 + 4);
    metadata.push(num_els);
    metadata.push(num_dims);
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
    // buffer(4): metadata
    set_params!(encoder, (&input, &indices, &values, output, metadata.as_slice()));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(indices.buffer, MTLResourceUsage::Read);
    encoder.use_resource(values.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes a gather operation to collect values from input using an index tensor along a dimension.
///
/// # Arguments
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `kernels` - Kernel cache
/// * `kernel_name` - Gather kernel (e.g., gather::F32)
/// * `input_shape` - Shape of input tensor
/// * `input` - Input tensor buffer (source data)
/// * `input_strides` - Strides of input tensor
/// * `input_offset` - Starting offset in input buffer
/// * `indices` - Index tensor buffer (determines which elements to gather)
/// * `indices_strides` - Strides of indices tensor
/// * `indices_offset` - Starting offset in indices buffer
/// * `dim` - Dimension along which to gather
/// * `output` - Output buffer (same shape as indices tensor)
///
/// # Metadata Layout
/// Total metadata length: `2 + num_dims * 3 + 4`
///
/// - `metadata[0]`: num_els (total output elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: input_shape
/// - `metadata[2+num_dims..2+2*num_dims]`: input_strides
/// - `metadata[2+2*num_dims..2+3*num_dims]`: indices_strides
/// - `metadata[2+3*num_dims]`: input_offset
/// - `metadata[2+3*num_dims+1]`: indices_offset
/// - `metadata[2+3*num_dims+2]`: dim (dimension to gather along)
/// - `metadata[2+3*num_dims+3]`: num_indices
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
    let mut metadata = Vec::with_capacity(2 + num_dims * 3 + 4);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend_from_slice(input_shape);
    metadata.extend_from_slice(input_strides);
    metadata.extend_from_slice(indices_strides);
    metadata.push(input_offset);
    metadata.push(indices_offset);
    metadata.push(dim);
    metadata.push(num_els); // num_indices = total number of indices

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): indices
    // buffer(2): output
    // buffer(3): metadata
    set_params!(encoder, (&input, &indices, output, metadata.as_slice()));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(indices.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes a scatter operation (scatter, scatter_add, scatter_max, scatter_min) to write source values
/// into output at positions specified by indices along a dimension.
///
/// # Arguments
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `kernels` - Kernel cache
/// * `kernel_name` - Scatter kernel (scatter::F32, scatter_add::F32, scatter_max::F32, scatter_min::F32)
/// * `input_shape` - Shape of input tensor (base tensor to scatter into)
/// * `input` - Input tensor buffer
/// * `input_strides` - Strides of input tensor
/// * `input_offset` - Starting offset in input buffer
/// * `indices` - Index tensor buffer (specifies where to scatter)
/// * `indices_strides` - Strides of indices tensor
/// * `indices_offset` - Starting offset in indices buffer
/// * `src` - Source tensor buffer (values to scatter)
/// * `src_shape` - Shape of source tensor
/// * `src_strides` - Strides of source tensor
/// * `src_offset` - Starting offset in source buffer
/// * `dim` - Dimension along which to scatter
/// * `output` - Output buffer (modified copy of input with scattered values)
///
/// # Metadata Layout
/// Total metadata length: `2 + num_dims * 5 + 4`
///
/// - `metadata[0]`: num_els (total source elements to scatter)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: input_shape
/// - `metadata[2+num_dims..2+2*num_dims]`: input_strides
/// - `metadata[2+2*num_dims..2+3*num_dims]`: src_shape
/// - `metadata[2+3*num_dims..2+4*num_dims]`: src_strides
/// - `metadata[2+4*num_dims..2+5*num_dims]`: indices_strides
/// - `metadata[2+5*num_dims]`: input_offset
/// - `metadata[2+5*num_dims+1]`: src_offset
/// - `metadata[2+5*num_dims+2]`: indices_offset
/// - `metadata[2+5*num_dims+3]`: dim (dimension to scatter along)
///
/// # Scatter Variants
/// - `scatter`: Overwrites values at indices
/// - `scatter_add`: Adds source values to existing values at indices
/// - `scatter_max`: Takes maximum of source and existing values at indices
/// - `scatter_min`: Takes minimum of source and existing values at indices
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
    let mut metadata = Vec::with_capacity(2 + num_dims * 5 + 4);
    metadata.push(num_els);
    metadata.push(num_dims);
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
    // buffer(4): metadata
    set_params!(encoder, (&input, &indices, &src, output, metadata.as_slice()));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(indices.buffer, MTLResourceUsage::Read);
    encoder.use_resource(src.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
