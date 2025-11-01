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

/// Executes a concatenation operation on multiple input tensors along a specified dimension.
///
/// # Arguments
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `kernels` - Kernel cache
/// * `kernel_name` - Concat operation kernel (e.g., concat::F32)
/// * `input` - Combined input buffer containing all input tensors
/// * `output` - Output buffer for concatenated result
/// * `metadata` - Metadata describing tensor shapes, strides, and offsets
///
/// # Metadata Layout
/// The metadata buffer must contain the following elements in order:
/// - `metadata[0]`: num_els (total number of output elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: output_shape (shape of output tensor)
/// - `metadata[2+num_dims]`: concat_dim (dimension along which to concatenate)
/// - `metadata[2+num_dims+1]`: num_inputs (number of input tensors)
/// - `metadata[2+num_dims+2..2+num_dims+2+num_inputs*num_dims]`: input_shapes (flattened)
/// - `metadata[2+num_dims+2+num_inputs*num_dims..2+num_dims+2+2*num_inputs*num_dims]`: input_strides (flattened)
/// - `metadata[2+num_dims+2+2*num_inputs*num_dims..2+num_dims+2+2*num_inputs*num_dims+num_inputs]`: input_offsets
/// - `metadata[2+num_dims+2+2*num_inputs*num_dims+num_inputs..]`: input_buffer_offsets
///
/// Total metadata length: `2 + num_dims + 2 + num_inputs * (2 * num_dims + 2)`
///
/// # Example
/// ```ignore
/// // Concatenate two 2x2 matrices along dimension 0 to form 4x2
/// let metadata = vec![
///     8,      // num_els
///     2,      // num_dims
///     4, 2,   // output_shape
///     0,      // concat_dim
///     2,      // num_inputs
///     2, 2,   // input_shapes[0]
///     2, 2,   // input_shapes[1]
///     2, 1,   // input_strides[0]
///     2, 1,   // input_strides[1]
///     0,      // input_offsets[0]
///     4,      // input_offsets[1]
///     0, 0,   // input_buffer_offsets
/// ];
/// call_concat(&device, &command_buffer, &kernels, concat::F32,
///             input_buffer, &output, &metadata)?;
/// ```
pub fn call_concat(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::ConcatSplit, kernel_name.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input (combined buffer with all inputs)
    // buffer(1): output
    // buffer(2): metadata
    set_params!(encoder, (&input, output, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes a split operation to extract a portion of a tensor along a specified dimension.
///
/// # Arguments
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `kernels` - Kernel cache
/// * `kernel_name` - Split operation kernel (e.g., split::F32)
/// * `input` - Input tensor buffer
/// * `output` - Output buffer for extracted portion
/// * `metadata` - Metadata describing tensor shape, strides, and split parameters
///
/// # Metadata Layout
/// The metadata buffer must contain the following elements in order:
/// - `metadata[0]`: num_els (total number of output elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: input_shape (shape of input tensor)
/// - `metadata[2+num_dims..2+2*num_dims]`: strides (stride for each dimension)
/// - `metadata[2+2*num_dims]`: offset (starting offset in input buffer)
/// - `metadata[2+2*num_dims+1]`: split_dim (dimension along which to split)
/// - `metadata[2+2*num_dims+2]`: output_size_on_dim (size of output on split dimension)
/// - `metadata[2+2*num_dims+3]`: split_offset (offset on split dimension)
///
/// Total metadata length: `2 + num_dims * 2 + 4`
///
/// # Example
/// ```ignore
/// // Split 4x2 matrix along dimension 0, taking 2 rows starting from offset 0
/// let metadata = vec![
///     4,      // num_els
///     2,      // num_dims
///     4, 2,   // input_shape
///     2, 1,   // strides
///     0,      // offset
///     0,      // split_dim
///     2,      // output_size_on_dim
///     0,      // split_offset
/// ];
/// call_split(&device, &command_buffer, &kernels, split::F32,
///            input_buffer, &output, &metadata)?;
/// ```
pub fn call_split(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::ConcatSplit, kernel_name.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): output
    // buffer(2): metadata
    set_params!(encoder, (&input, output, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
