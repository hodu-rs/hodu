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

/// Executes a convolution operation (1D, 2D, 3D, or transposed variants) using Metal compute pipeline.
///
/// # Arguments
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `kernels` - Kernel cache
/// * `kernel_name` - Convolution kernel (conv1d::F32, conv2d::F32, conv3d::F32, conv_transpose1d::F32, etc.)
/// * `input` - Input tensor buffer (batch, in_channels, spatial_dims...)
/// * `weight` - Weight tensor buffer (out_channels, in_channels, kernel_size...)
/// * `output` - Output buffer for convolution result
/// * `metadata` - Metadata describing convolution parameters
///
/// # Metadata Layout
///
/// ## Conv1D
/// - `metadata[0]`: num_els (total output elements)
/// - `metadata[1]`: batch
/// - `metadata[2]`: in_channels
/// - `metadata[3]`: out_channels
/// - `metadata[4]`: input_length
/// - `metadata[5]`: kernel_size
/// - `metadata[6]`: output_length
/// - `metadata[7]`: stride
/// - `metadata[8]`: padding
/// - `metadata[9]`: dilation
/// - `metadata[10]`: input_offset
/// - `metadata[11]`: weight_offset
///
/// ## Conv2D
/// - `metadata[0]`: num_els
/// - `metadata[1]`: batch
/// - `metadata[2]`: in_channels
/// - `metadata[3]`: out_channels
/// - `metadata[4]`: input_height
/// - `metadata[5]`: input_width
/// - `metadata[6]`: kernel_height
/// - `metadata[7]`: kernel_width
/// - `metadata[8]`: output_height
/// - `metadata[9]`: output_width
/// - `metadata[10]`: stride_h
/// - `metadata[11]`: stride_w
/// - `metadata[12]`: padding_h
/// - `metadata[13]`: padding_w
/// - `metadata[14]`: dilation_h
/// - `metadata[15]`: dilation_w
/// - `metadata[16]`: input_offset
/// - `metadata[17]`: weight_offset
///
/// ## Conv3D
/// - `metadata[0]`: num_els
/// - `metadata[1]`: batch
/// - `metadata[2]`: in_channels
/// - `metadata[3]`: out_channels
/// - `metadata[4]`: input_depth
/// - `metadata[5]`: input_height
/// - `metadata[6]`: input_width
/// - `metadata[7]`: kernel_depth
/// - `metadata[8]`: kernel_height
/// - `metadata[9]`: kernel_width
/// - `metadata[10]`: output_depth
/// - `metadata[11]`: output_height
/// - `metadata[12]`: output_width
/// - `metadata[13]`: stride_d
/// - `metadata[14]`: stride_h
/// - `metadata[15]`: stride_w
/// - `metadata[16]`: padding_d
/// - `metadata[17]`: padding_h
/// - `metadata[18]`: padding_w
/// - `metadata[19]`: dilation_d
/// - `metadata[20]`: dilation_h
/// - `metadata[21]`: dilation_w
/// - `metadata[22]`: input_offset
/// - `metadata[23]`: weight_offset
///
/// Transpose convolutions have similar layouts with additional output_padding parameters.
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

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): weight
    // buffer(2): output
    // buffer(3): metadata (num_els, ...)
    set_params!(encoder, (&input, &weight, output, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(weight.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes a convolution weight gradient operation for backpropagation using Metal compute pipeline.
///
/// # Arguments
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `kernels` - Kernel cache
/// * `kernel_name` - Gradient kernel (conv1d_grad_weight::F32, conv2d_grad_weight::F32, etc.)
/// * `input` - Input tensor buffer from forward pass
/// * `grad_output` - Gradient from next layer (upstream gradient)
/// * `grad_weight` - Output buffer for weight gradients
/// * `metadata` - Metadata describing gradient computation parameters
///
/// # Metadata Layout
///
/// ## Conv1D Grad Weight
/// - `metadata[0]`: num_els (total grad_weight elements)
/// - `metadata[1]`: batch
/// - `metadata[2]`: in_channels
/// - `metadata[3]`: out_channels
/// - `metadata[4]`: input_length
/// - `metadata[5]`: kernel_width
/// - `metadata[6]`: output_length
/// - `metadata[7]`: stride
/// - `metadata[8]`: padding
/// - `metadata[9]`: dilation
/// - `metadata[10]`: input_offset
/// - `metadata[11]`: grad_output_offset
///
/// ## Conv2D Grad Weight
/// - `metadata[0]`: num_els
/// - `metadata[1]`: batch
/// - `metadata[2]`: in_channels
/// - `metadata[3]`: out_channels
/// - `metadata[4]`: input_height
/// - `metadata[5]`: input_width
/// - `metadata[6]`: kernel_height
/// - `metadata[7]`: kernel_width
/// - `metadata[8]`: output_height
/// - `metadata[9]`: output_width
/// - `metadata[10]`: stride_h
/// - `metadata[11]`: stride_w
/// - `metadata[12]`: padding_h
/// - `metadata[13]`: padding_w
/// - `metadata[14]`: dilation_h
/// - `metadata[15]`: dilation_w
/// - `metadata[16]`: input_offset
/// - `metadata[17]`: grad_output_offset
///
/// ## Conv3D Grad Weight
/// - `metadata[0]`: num_els
/// - `metadata[1]`: batch
/// - `metadata[2]`: in_channels
/// - `metadata[3]`: out_channels
/// - `metadata[4]`: input_depth
/// - `metadata[5]`: input_height
/// - `metadata[6]`: input_width
/// - `metadata[7]`: kernel_depth
/// - `metadata[8]`: kernel_height
/// - `metadata[9]`: kernel_width
/// - `metadata[10]`: output_depth
/// - `metadata[11]`: output_height
/// - `metadata[12]`: output_width
/// - `metadata[13]`: stride_d
/// - `metadata[14]`: stride_h
/// - `metadata[15]`: stride_w
/// - `metadata[16]`: padding_d
/// - `metadata[17]`: padding_h
/// - `metadata[18]`: padding_w
/// - `metadata[19]`: dilation_d
/// - `metadata[20]`: dilation_h
/// - `metadata[21]`: dilation_w
/// - `metadata[22]`: input_offset
/// - `metadata[23]`: grad_output_offset
///
/// Transpose convolutions have similar layouts with additional output_padding parameters.
///
/// Note: These operations use atomic operations for parallel reduction across batch and spatial dimensions.
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

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): grad_output
    // buffer(2): grad_weight
    // buffer(3): metadata (num_els, ...)
    set_params!(encoder, (&input, &grad_output, grad_weight, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(grad_output.buffer, MTLResourceUsage::Read);
    encoder.use_resource(grad_weight, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
