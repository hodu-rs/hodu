use crate::{
    error::MetalKernelError,
    kernel::Kernels,
    kernels::macros::ops,
    metal::{Buffer, ComputeCommandEncoder, Device},
    set_params,
    source::Source,
    utils::{linear_split, EncoderParam, EncoderProvider},
};
use objc2_metal::MTLResourceUsage;

ops!(const_set);

/// Executes a constant fill operation to set all elements of a tensor to a constant value.
///
/// This operation fills a tensor (possibly with strided layout) with a single constant value.
/// It supports non-contiguous layouts, so only the logical tensor elements are modified,
/// leaving gaps in strided buffers untouched.
///
/// # Arguments
/// * `kernel` - Const set kernel (e.g., const_set::F32)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `output` - Output buffer (will be filled with const_val)
/// * `metadata` - Metadata describing tensor shape, strides, and offset
/// * `const_val` - Constant value to fill the tensor with
///
/// # Metadata Layout
/// Total metadata length: `2 + num_dims * 2 + 1`
///
/// - `metadata[0]`: num_els (total number of elements to set)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: shape (dimensions of the tensor)
/// - `metadata[2+num_dims..2+2*num_dims]`: strides (stride for each dimension)
/// - `metadata[2+2*num_dims]`: offset (starting offset in output buffer)
///
/// # Type Parameter
/// * `T: EncoderParam` - The type of the constant value (f32, i32, bool, etc.)
///
/// # Example
/// ```ignore
/// // Fill a 3x4 matrix with value 7.0
/// let metadata = vec![
///     12,     // num_els (3 * 4)
///     2,      // num_dims
///     3, 4,   // shape
///     4, 1,   // strides (row-major)
///     0,      // offset
/// ];
/// call_const_set(&device, &command_buffer, &kernels, const_set::F32,
///                &output, &metadata, 7.0f32)?;
/// ```
///
/// # Example with Strided Layout
/// ```ignore
/// // Fill a 2x3 tensor with non-contiguous strides
/// // Only positions [0, 2, 4, 6, 8, 10] will be set to 9.0
/// let metadata = vec![
///     6,      // num_els (2 * 3)
///     2,      // num_dims
///     2, 3,   // shape
///     6, 2,   // strides (non-contiguous)
///     0,      // offset
/// ];
/// call_const_set(&device, &command_buffer, &kernels, const_set::F32,
///                &output, &metadata, 9.0f32)?;
/// ```
pub fn call_const_set<T: EncoderParam>(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    output: &Buffer,
    metadata: &[usize],
    const_val: T,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Storage, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): output
    // buffer(1): metadata
    // buffer(2): const_val
    set_params!(encoder, (output, metadata, const_val));

    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
