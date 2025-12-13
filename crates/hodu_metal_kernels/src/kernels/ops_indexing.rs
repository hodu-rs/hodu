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
    scatter_min,
    onehot,
    nonzero_count,
    nonzero_fill,
    unique_sort,
    unique_bitonic_step,
    unique_count,
    unique_mark,
    unique_build
);

/// Executes an index_select operation to extract elements along a dimension using indices.
///
/// # Arguments
/// * `kernel` - Index select kernel (e.g., index_select::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer
/// * `indices` - Index tensor buffer (contains indices to select)
/// * `output` - Output buffer
/// * `metadata` - Metadata describing tensor layout and operation parameters
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
/// # Kernel signature
/// `(input, indices, output, metadata)`
#[allow(clippy::too_many_arguments)]
pub fn call_ops_index_select(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    indices: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): indices
    // buffer(2): output
    // buffer(3): metadata
    set_params!(encoder, (&input, &indices, output, metadata));

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
/// * `kernel` - Index put kernel (e.g., index_put::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer
/// * `indices` - Index tensor buffer (specifies where to put values)
/// * `values` - Values tensor buffer (values to write)
/// * `output` - Output buffer (modified copy of input)
/// * `metadata` - Metadata describing tensor layout and operation parameters
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
///
/// # Kernel signature
/// `(input, indices, values, output, metadata)`
#[allow(clippy::too_many_arguments)]
pub fn call_ops_index_put(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    indices: BufferOffset,
    values: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): indices
    // buffer(2): values
    // buffer(3): output
    // buffer(4): metadata
    set_params!(encoder, (&input, &indices, &values, output, metadata));

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
/// * `kernel` - Gather kernel (e.g., gather::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer (source data)
/// * `indices` - Index tensor buffer (determines which elements to gather)
/// * `output` - Output buffer
/// * `metadata` - Metadata describing tensor layout and operation parameters
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
///
/// # Kernel signature
/// `(input, indices, output, metadata)`
#[allow(clippy::too_many_arguments)]
pub fn call_ops_gather(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    indices: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): indices
    // buffer(2): output
    // buffer(3): metadata
    set_params!(encoder, (&input, &indices, output, metadata));

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
/// * `kernel` - Scatter kernel (scatter::F32, scatter_add::F32, scatter_max::F32, scatter_min::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer
/// * `indices` - Index tensor buffer (specifies where to scatter)
/// * `src` - Source tensor buffer (values to scatter)
/// * `output` - Output buffer (modified copy of input with scattered values)
/// * `metadata` - Metadata describing tensor layout and operation parameters
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
/// # Kernel signature
/// `(input, indices, src, output, metadata)`
///
/// # Scatter Variants
/// - `scatter`: Overwrites values at indices
/// - `scatter_add`: Adds source values to existing values at indices
/// - `scatter_max`: Takes maximum of source and existing values at indices
/// - `scatter_min`: Takes minimum of source and existing values at indices
#[allow(clippy::too_many_arguments)]
pub fn call_ops_scatter(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    indices: BufferOffset,
    src: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): indices
    // buffer(2): src
    // buffer(3): output
    // buffer(4): metadata
    set_params!(encoder, (&input, &indices, &src, output, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(indices.buffer, MTLResourceUsage::Read);
    encoder.use_resource(src.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes a onehot operation to convert integer indices to one-hot encoded vectors.
///
/// # Arguments
/// * `kernel` - Onehot kernel (e.g., onehot::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `indices` - Index tensor buffer (int32 indices)
/// * `output` - Output buffer (will contain one-hot encoded values)
/// * `metadata` - Metadata describing tensor layout and operation parameters
///
/// # Metadata Layout
/// Total metadata length: `5 + num_dims_out`
///
/// - `metadata[0]`: num_els (total output elements)
/// - `metadata[1]`: num_input_els (total input elements)
/// - `metadata[2]`: num_classes (depth of one-hot dimension)
/// - `metadata[3]`: axis (dimension for one-hot encoding)
/// - `metadata[4]`: num_dims_out (number of output dimensions)
/// - `metadata[5..5+num_dims_out]`: output_shape
///
/// # Kernel signature
/// `(indices, output, metadata)`
#[allow(clippy::too_many_arguments)]
pub fn call_ops_onehot(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    indices: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): indices
    // buffer(1): output
    // buffer(2): metadata
    set_params!(encoder, (&indices, output, metadata));

    encoder.use_resource(indices.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes nonzero count operation to count non-zero elements.
#[allow(clippy::too_many_arguments)]
pub fn call_nonzero_count(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    count_buffer: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (&input, count_buffer, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(count_buffer, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes nonzero fill operation to fill indices of non-zero elements.
#[allow(clippy::too_many_arguments)]
pub fn call_nonzero_fill(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    counter_buffer: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (&input, output, counter_buffer, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.use_resource(counter_buffer, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes unique sort operation to copy and initialize indices.
#[allow(clippy::too_many_arguments)]
pub fn call_unique_sort(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    sorted_values: &Buffer,
    sorted_indices: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel.0)?;

    // metadata[2] is padded_size (next power of 2 for bitonic sort)
    let padded_size = metadata[2];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (&input, sorted_values, sorted_indices, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(sorted_values, MTLResourceUsage::Write);
    encoder.use_resource(sorted_indices, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, padded_size);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes bitonic sort step for unique operation.
#[allow(clippy::too_many_arguments)]
pub fn call_unique_bitonic_step(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    values: &Buffer,
    indices: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel.0)?;

    // metadata layout: [num_els, offset, padded_size, k, j]
    let padded_size = metadata[2];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (values, indices, metadata));

    encoder.use_resource(values, MTLResourceUsage::Read | MTLResourceUsage::Write);
    encoder.use_resource(indices, MTLResourceUsage::Read | MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, padded_size);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes unique count operation to count unique elements.
#[allow(clippy::too_many_arguments)]
pub fn call_unique_count(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    sorted_values: &Buffer,
    count_buffer: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (sorted_values, count_buffer, metadata));

    encoder.use_resource(sorted_values, MTLResourceUsage::Read);
    encoder.use_resource(count_buffer, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes unique mark operation to mark unique boundaries.
#[allow(clippy::too_many_arguments)]
pub fn call_unique_mark(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    sorted_values: &Buffer,
    marks: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (sorted_values, marks, metadata));

    encoder.use_resource(sorted_values, MTLResourceUsage::Read);
    encoder.use_resource(marks, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes unique prefix sum operation (single threaded).
#[allow(clippy::too_many_arguments)]
pub fn call_unique_prefix_sum(
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    marks: &Buffer,
    unique_idx: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, "hodu_metal_unique_prefix_sum")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (marks, unique_idx, metadata));

    encoder.use_resource(marks, MTLResourceUsage::Read);
    encoder.use_resource(unique_idx, MTLResourceUsage::Write);

    // Single thread for sequential prefix sum
    encoder.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );

    Ok(())
}

/// Executes unique build operation to construct output arrays.
#[allow(clippy::too_many_arguments)]
pub fn call_unique_build(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    sorted_values: &Buffer,
    sorted_indices: &Buffer,
    marks: &Buffer,
    unique_idx: &Buffer,
    values: &Buffer,
    inverse: &Buffer,
    counts: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Indexing, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            sorted_values,
            sorted_indices,
            marks,
            unique_idx,
            values,
            inverse,
            counts,
            metadata
        )
    );

    encoder.use_resource(sorted_values, MTLResourceUsage::Read);
    encoder.use_resource(sorted_indices, MTLResourceUsage::Read);
    encoder.use_resource(marks, MTLResourceUsage::Read);
    encoder.use_resource(unique_idx, MTLResourceUsage::Read);
    encoder.use_resource(values, MTLResourceUsage::Write);
    encoder.use_resource(inverse, MTLResourceUsage::Write);
    encoder.use_resource(counts, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
