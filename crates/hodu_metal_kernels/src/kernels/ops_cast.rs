use crate::{
    error::MetalKernelError,
    kernel::Kernels,
    kernels::Kernel,
    metal::{Buffer, ComputeCommandEncoder, Device},
    set_params,
    source::Source,
    utils::{linear_split, BufferOffset, EncoderProvider},
};
use objc2_metal::MTLResourceUsage;

/// Executes a type cast operation on an input tensor using Metal compute pipeline.
///
/// # Arguments
/// * `kernel` - Cast operation kernel name (e.g., "hodu_metal_cast_f32_to_i32", "hodu_metal_cast_i32_to_f32")
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer with offset
/// * `output` - Output buffer (must be sized appropriately for output type)
/// * `metadata` - Metadata describing tensor shape and layout
///
/// # Metadata Layout
/// The metadata buffer must contain the following elements in order:
/// - `metadata[0]`: num_els (total number of elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: shape (dimensions of the tensor)
/// - `metadata[2+num_dims..2+2*num_dims]`: strides (stride for each dimension)
/// - `metadata[2+2*num_dims]`: offset (starting offset in input buffer)
///
/// Total metadata length: `2 + num_dims * 2 + 1`
///
/// # Example
/// ```ignore
/// let metadata = vec![
///     4,      // num_els
///     1,      // num_dims
///     4,      // shape[0]
///     1,      // strides[0]
///     0,      // offset
/// ];
/// call_ops_cast(&device, &command_buffer, &kernels, "hodu_metal_cast_f32_to_i32",
///           input_buffer, &output, &metadata)?;
/// ```
pub fn call_ops_cast(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Cast, kernel.0)?;

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
