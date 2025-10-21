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

#[allow(clippy::too_many_arguments)]
pub fn call_binary(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    shape: &[usize],
    left_input: BufferOffset,
    left_strides: &[usize],
    left_offset: usize,
    right_input: BufferOffset,
    right_strides: &[usize],
    right_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Binary, kernel_name.0)?;

    let num_dims = shape.len();
    let num_els: usize = shape.iter().product();

    // Prepare metadata: dims, lhs_strides, rhs_strides, lhs_offset, rhs_offset
    let mut metadata = Vec::with_capacity(num_dims * 3 + 2);
    metadata.extend_from_slice(shape);
    metadata.extend_from_slice(left_strides);
    metadata.extend_from_slice(right_strides);
    metadata.push(left_offset);
    metadata.push(right_offset);

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): lhs input
    // buffer(1): rhs input
    // buffer(2): output
    // buffer(3): num_els
    // buffer(4): num_dims
    // buffer(5): metadata (dims, lhs_strides, rhs_strides, lhs_offset, rhs_offset)
    set_params!(
        encoder,
        (
            &left_input,
            &right_input,
            output,
            num_els,
            num_dims,
            metadata.as_slice()
        )
    );

    encoder.use_resource(left_input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(right_input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
