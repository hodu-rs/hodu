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
    add,
    sub,
    mul,
    div,
    pow,
    minimum,
    maximum,
    eq,
    ne,
    le,
    lt,
    ge,
    gt,
    logical_and,
    logical_or,
    logical_xor
);

/// Executes a binary operation on two input tensors using Metal compute pipeline.
///
/// # Arguments
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `kernels` - Kernel cache
/// * `kernel_name` - Binary operation kernel to execute (add, sub, mul, div, etc.)
/// * `left_input` - Left input tensor buffer with offset
/// * `right_input` - Right input tensor buffer with offset
/// * `output` - Output buffer
/// * `metadata` - Metadata describing tensor shapes and strides
///
/// # Metadata Layout
/// The metadata buffer must contain the following elements in order:
/// - `metadata[0]`: num_els (total number of output elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: lhs_shape (shape of left input)
/// - `metadata[2+num_dims..2+2*num_dims]`: rhs_shape (shape of right input)
/// - `metadata[2+2*num_dims..2+3*num_dims]`: lhs_strides (strides of left input)
/// - `metadata[2+3*num_dims..2+4*num_dims]`: rhs_strides (strides of right input)
/// - `metadata[2+4*num_dims]`: lhs_offset (starting offset in left input buffer)
/// - `metadata[2+4*num_dims+1]`: rhs_offset (starting offset in right input buffer)
///
/// Total metadata length: `2 + num_dims * 4 + 2`
///
/// # Example
/// ```ignore
/// let shape = vec![3];
/// let strides = vec![1];
/// let metadata = vec![
///     3,      // num_els
///     1,      // num_dims
///     3,      // lhs_shape[0]
///     3,      // rhs_shape[0]
///     1,      // lhs_strides[0]
///     1,      // rhs_strides[0]
///     0,      // lhs_offset
///     0,      // rhs_offset
/// ];
/// call_binary(&device, &command_buffer, &kernels, add::F32,
///             lhs_buffer, rhs_buffer, &output, &metadata)?;
/// ```
#[allow(clippy::too_many_arguments)]
pub fn call_binary(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    left_input: BufferOffset,
    right_input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Binary, kernel_name.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): lhs input
    // buffer(1): rhs input
    // buffer(2): output
    // buffer(3): metadata
    set_params!(encoder, (&left_input, &right_input, output, metadata));

    encoder.use_resource(left_input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(right_input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
