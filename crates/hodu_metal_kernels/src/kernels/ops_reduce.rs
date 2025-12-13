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

ops!(sum, max, min, prod, mean, norm, logsum, logsumexp, argmax, argmin, any, all);

/// Executes a reduction operation along specified dimensions using Metal compute pipeline.
///
/// Reduces a tensor along one or more dimensions using operations like sum, max, min, mean, etc.
/// The reduction can either keep the reduced dimensions (with size 1) or remove them entirely.
///
/// # Arguments
/// * `kernel` - Reduce kernel (reduce_sum::F32, reduce_max::F32, reduce_min::F32, reduce_mean::F32,
///   reduce_argmax::F32, reduce_argmin::F32, reduce_prod::F32, reduce_norm::F32, etc.)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer
/// * `output` - Output buffer
/// * `metadata` - Metadata describing tensor layout and operation parameters
///
/// # Metadata Layout
/// Variable length based on tensor dimensionality and number of reduce dimensions:
///
/// - `metadata[0]`: num_dims (number of input dimensions)
/// - `metadata[1..1+num_dims]`: shape (input tensor shape)
/// - `metadata[1+num_dims..1+2*num_dims]`: strides (input tensor strides)
/// - `metadata[1+2*num_dims]`: offset (starting offset in input)
/// - `metadata[2+2*num_dims]`: output_shape_len (number of output dimensions)
/// - `metadata[3+2*num_dims..3+2*num_dims+output_shape_len]`: output_shape
/// - `metadata[3+2*num_dims+output_shape_len]`: num_reduce_dims (number of dimensions to reduce)
/// - `metadata[4+2*num_dims+output_shape_len..4+2*num_dims+output_shape_len+num_reduce_dims]`: reduce_dims
/// - `metadata[4+2*num_dims+output_shape_len+num_reduce_dims]`: keep_dim (1 if true, 0 if false)
/// - `metadata[4+2*num_dims+output_shape_len+num_reduce_dims+1]`: reduce_size
///
/// # Kernel signature
/// `(input, output, metadata)`
///
/// # Keep_dim behavior
/// - If keep_dim=true: output shape matches input but reduced dims have size 1
/// - If keep_dim=false: reduced dimensions are squeezed out of output
///
/// # Supported Operations
/// - `sum`: Sum of elements
/// - `max`: Maximum element
/// - `min`: Minimum element
/// - `mean`: Mean (average) of elements
/// - `prod`: Product of elements
/// - `norm`: L2 norm
/// - `argmax`: Index of maximum element (returns i32)
/// - `argmin`: Index of minimum element (returns i32)
/// - `any`: Logical OR (for boolean tensors)
/// - `all`: Logical AND (for boolean tensors)
pub fn call_ops_reduce(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel.0)?;

    // Calculate num_els from output_shape in metadata
    let num_dims = metadata[0];
    let output_shape_len = metadata[2 + 2 * num_dims];
    let output_shape_start = 3 + 2 * num_dims;
    let num_els: usize = metadata[output_shape_start..output_shape_start + output_shape_len]
        .iter()
        .product();

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
