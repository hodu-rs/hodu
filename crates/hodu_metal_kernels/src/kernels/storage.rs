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

#[allow(clippy::too_many_arguments)]
pub fn call_const_set<T: EncoderParam>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    const_val: T,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Storage, kernel_name.0)?;

    let num_dims = shape.len();
    let num_els: usize = shape.iter().product();

    // Prepare metadata: dims, strides, offset
    let mut metadata = Vec::with_capacity(num_dims * 2 + 1);
    metadata.extend_from_slice(shape);
    metadata.extend_from_slice(strides);
    metadata.push(offset);

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): output
    // buffer(1): num_els
    // buffer(2): num_dims
    // buffer(3): metadata (dims, strides, offset)
    // buffer(4): const_val
    set_params!(encoder, (output, num_els, num_dims, metadata.as_slice(), const_val));

    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
