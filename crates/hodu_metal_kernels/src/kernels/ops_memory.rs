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

ops!(contiguous, copy);

/// Executes a contiguous operation to convert a strided tensor layout to a contiguous layout.
///
/// This operation reads from a potentially non-contiguous (strided) tensor and writes
/// the elements in a contiguous, row-major order. This is useful for operations that
/// require contiguous memory layout or to materialize views (e.g., transposes, slices).
///
/// # Arguments
/// * `kernel` - Contiguous kernel (e.g., contiguous::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer (may have non-contiguous strides)
/// * `output` - Output buffer (will be contiguous)
/// * `metadata` - Metadata describing tensor shape, strides, and offset
///
/// # Metadata Layout
/// Total metadata length: `2 + num_dims * 2 + 1`
///
/// - `metadata[0]`: num_els (total number of elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: shape (dimensions of the tensor)
/// - `metadata[2+num_dims..2+2*num_dims]`: strides (stride for each dimension)
/// - `metadata[2+2*num_dims]`: offset (starting offset in input buffer)
///
/// # Example
/// ```ignore
/// // Make a transposed 2x3 matrix contiguous
/// // Input (strided): [[1, 2, 3], [4, 5, 6]] with strides [1, 3] (column-major)
/// // Output (contiguous): [[1, 4], [2, 5], [3, 6]] with strides [2, 1] (row-major)
/// let metadata = vec![
///     6,      // num_els
///     2,      // num_dims
///     3, 2,   // shape (transposed)
///     1, 3,   // strides (column-major)
///     0,      // offset
/// ];
/// call_ops_contiguous(&device, &command_buffer, &kernels, contiguous::F32,
///                 input_buffer, &output, &metadata)?;
/// ```
pub fn call_ops_contiguous(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Memory, kernel.0)?;

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

/// Executes a simple element-wise copy operation from input buffer to output buffer.
///
/// This is a straightforward parallel copy operation that copies `num_els` elements
/// from the input buffer to the output buffer. Both buffers must be contiguous
/// and have the same element type.
///
/// # Arguments
/// * `kernel` - Copy kernel (e.g., copy::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `num_els` - Number of elements to copy
/// * `input` - Input tensor buffer (source)
/// * `output` - Output buffer (destination)
///
/// # Parameters
/// No complex metadata required - only `num_els` is passed to the kernel.
///
/// # Performance
/// This is a memory-bandwidth-bound operation. For large tensors, consider
/// using Metal's native buffer copy operations (`blit` command encoder) which
/// may be more efficient.
///
/// # Example
/// ```ignore
/// // Copy 1000 f32 elements
/// call_ops_copy(&device, &command_buffer, &kernels, copy::F32,
///           1000, input_buffer, &output)?;
/// ```
pub fn call_ops_copy(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    num_els: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Memory, kernel.0)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(encoder, (&input, output, num_els));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
