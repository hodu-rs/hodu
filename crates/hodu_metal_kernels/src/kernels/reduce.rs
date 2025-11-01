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

/// Executes a reduction operation along specified dimensions using Metal compute pipeline.
///
/// Reduces a tensor along one or more dimensions using operations like sum, max, min, mean, etc.
/// The reduction can either keep the reduced dimensions (with size 1) or remove them entirely.
///
/// # Arguments
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `kernels` - Kernel cache
/// * `kernel_name` - Reduce kernel (reduce_sum::F32, reduce_max::F32, reduce_min::F32, reduce_mean::F32,
///                   reduce_argmax::F32, reduce_argmin::F32, reduce_prod::F32, reduce_norm::F32, etc.)
/// * `shape` - Shape of input tensor
/// * `input` - Input tensor buffer
/// * `input_strides` - Strides of input tensor
/// * `input_offset` - Starting offset in input buffer
/// * `reduce_dims` - Dimensions to reduce along (e.g., [1] to reduce along dimension 1)
/// * `reduce_size` - Total number of elements being reduced (product of sizes of reduce_dims)
/// * `keep_dim` - If true, reduced dimensions have size 1; if false, they are removed
/// * `output` - Output buffer
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
/// Total metadata length: `1 + num_dims * 2 + 1 + 1 + output_shape_len + 1 + num_reduce_dims + 2`
///
/// # Example
/// ```ignore
/// // Reduce 2x3 matrix along dimension 1 (sum columns)
/// // Input: [[1, 2, 3], [4, 5, 6]] -> Output: [6, 15]
/// let shape = vec![2, 3];
/// let strides = vec![3, 1];
/// let reduce_dims = vec![1];
/// let reduce_size = 3; // reducing 3 elements per output
///
/// call_reduce(
///     &device, &command_buffer, &kernels, reduce_sum::F32,
///     &shape, input_buffer, &strides, 0, &reduce_dims, reduce_size, false, &output
/// )?;
/// ```
///
/// # Supported Operations
/// - `reduce_sum`: Sum of elements
/// - `reduce_max`: Maximum element
/// - `reduce_min`: Minimum element
/// - `reduce_mean`: Mean (average) of elements
/// - `reduce_prod`: Product of elements
/// - `reduce_norm`: L2 norm
/// - `reduce_argmax`: Index of maximum element (returns i32)
/// - `reduce_argmin`: Index of minimum element (returns i32)
/// - `reduce_any`: Logical OR (for boolean tensors)
/// - `reduce_all`: Logical AND (for boolean tensors)
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
    reduce_dims: &[usize],
    reduce_size: usize,
    keep_dim: bool,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name.0)?;

    let num_dims = shape.len();
    let num_reduce_dims = reduce_dims.len();

    // Calculate output shape based on keep_dim
    let mut output_shape = shape.to_vec();
    for &dim in reduce_dims {
        if keep_dim {
            output_shape[dim] = 1;
        } else {
            output_shape[dim] = 0; // Mark for removal
        }
    }
    if !keep_dim {
        output_shape.retain(|&size| size != 0);
        if output_shape.is_empty() {
            output_shape = vec![1]; // Scalar result
        }
    }

    let num_els: usize = output_shape.iter().product();
    let output_shape_len = output_shape.len();
    let mut metadata = Vec::with_capacity(1 + num_dims * 2 + 1 + 1 + output_shape_len + 1 + num_reduce_dims + 2);
    metadata.push(num_dims);
    metadata.extend_from_slice(shape);
    metadata.extend_from_slice(input_strides);
    metadata.push(input_offset);
    metadata.push(output_shape_len);
    metadata.extend_from_slice(&output_shape);
    metadata.push(num_reduce_dims);
    metadata.extend_from_slice(reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): output
    // buffer(2): metadata
    set_params!(encoder, (&input, output, metadata.as_slice()));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
