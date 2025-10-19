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
    conv1d,
    conv2d,
    conv3d,
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
    conv1d_grad_weight,
    conv2d_grad_weight,
    conv3d_grad_weight,
    conv_transpose1d_grad_weight,
    conv_transpose2d_grad_weight,
    conv_transpose3d_grad_weight
);

/// Call convolution operation (1D, 2D, or 3D)
#[allow(clippy::too_many_arguments)]
pub fn call_conv(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    input: BufferOffset,
    weight: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, kernel_name.0)?;

    let num_els = metadata[0]; // First element in metadata should be num_els

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): weight
    // buffer(2): output
    // buffer(3): num_els
    // buffer(4): metadata (varies by conv type)
    set_params!(encoder, (&input, &weight, output, num_els, &metadata[1..]));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(weight.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Call convolution gradient weight operation
#[allow(clippy::too_many_arguments)]
pub fn call_conv_grad_weight(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    input: BufferOffset,
    grad_output: BufferOffset,
    grad_weight: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, kernel_name.0)?;

    let num_els = metadata[0]; // First element in metadata should be num_els

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): grad_output
    // buffer(2): grad_weight
    // buffer(3): num_els
    // buffer(4): metadata (varies by conv type)
    set_params!(encoder, (&input, &grad_output, grad_weight, num_els, &metadata[1..]));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(grad_output.buffer, MTLResourceUsage::Read);
    encoder.use_resource(grad_weight, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
