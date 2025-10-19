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
    abs,
    neg,
    sign,
    square,
    sqrt,
    recip,
    relu,
    sigmoid,
    tanh,
    gelu,
    softplus,
    sin,
    cos,
    tan,
    exp,
    exp2,
    exp10,
    ln,
    log2,
    log10,
    logical_not,
    add_scalar,
    sub_scalar,
    mul_scalar,
    div_scalar,
    pow_scalar,
    maximum_scalar,
    minimum_scalar,
    eq_scalar,
    ne_scalar,
    lt_scalar,
    le_scalar,
    gt_scalar,
    ge_scalar,
    leaky_relu,
    elu
);

#[allow(clippy::too_many_arguments)]
pub fn call_unary(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    shape: &[usize],
    input: BufferOffset,
    input_strides: &[usize],
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, kernel_name.0)?;

    let num_dims = shape.len();
    let num_els: usize = shape.iter().product();

    // Prepare metadata: dims, strides, offset
    let mut metadata = Vec::with_capacity(num_dims * 2 + 1);
    metadata.extend_from_slice(shape);
    metadata.extend_from_slice(input_strides);
    metadata.push(input_offset);

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input (can be nullptr for in-place)
    // buffer(1): output
    // buffer(2): num_els
    // buffer(3): num_dims
    // buffer(4): metadata (dims, strides, offset)
    set_params!(encoder, (&input, output, num_els, num_dims, metadata.as_slice()));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
