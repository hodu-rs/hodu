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

ops!(resize);

/// Executes a resize operation (spatial interpolation) using Metal compute pipeline.
///
/// Resizes the spatial dimensions of the input tensor using various interpolation modes.
/// Supports ONNX-compatible coordinate transformation modes and nearest neighbor rounding modes.
///
/// # Arguments
/// * `kernel` - Resize kernel (resize::F32, resize::F16, resize::BF16)
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `input` - Input tensor buffer
/// * `output` - Output buffer for resized tensor
/// * `metadata` - Metadata describing tensor shapes and resize parameters
///
/// # Metadata Layout
/// - `metadata[0]`: output_size (total number of elements in output)
/// - `metadata[1]`: num_dims (number of dimensions, typically 4 for NCHW or 5 for NCDHW)
/// - `metadata[2..2+num_dims]`: input_shape
/// - `metadata[2+num_dims..2+2*num_dims]`: input_strides
/// - `metadata[2+2*num_dims]`: offset (starting offset in input)
/// - `metadata[3+2*num_dims..3+3*num_dims]`: output_shape
/// - `metadata[3+3*num_dims]`: mode (0=nearest, 1=linear, 2=cubic)
/// - `metadata[4+3*num_dims]`: coord_transform (0=half_pixel, 1=asymmetric, 2=align_corners, 3=pytorch_half_pixel)
/// - `metadata[5+3*num_dims]`: nearest_mode (0=floor, 1=ceil, 2=round_prefer_floor, 3=round_prefer_ceil)
///
/// # Interpolation Modes
/// - Nearest (0): Nearest neighbor interpolation (no backprop)
/// - Linear (1): Bilinear (2D) or trilinear (3D) interpolation
/// - Cubic (2): Bicubic interpolation (2D only)
///
/// # Coordinate Transformation Modes
/// - HalfPixel (0): `out_coord = (in_coord + 0.5) * scale - 0.5`
/// - Asymmetric (1): `out_coord = in_coord * scale`
/// - AlignCorners (2): `out_coord = in_coord * (in_size - 1) / (out_size - 1)`
/// - PytorchHalfPixel (3): Like HalfPixel but returns 0 when output size is 1
///
/// # Nearest Rounding Modes
/// - Floor (0): Round down
/// - Ceil (1): Round up
/// - RoundPreferFloor (2): Round to nearest, prefer floor on tie
/// - RoundPreferCeil (3): Round to nearest, prefer ceil on tie
///
/// # Example - 2D Bilinear Upsampling
/// ```ignore
/// // Input: [1, 1, 2, 2] (NCHW), Output: [1, 1, 4, 4]
/// let metadata = vec![
///     16,         // output_size (1*1*4*4)
///     4,          // num_dims
///     1, 1, 2, 2, // input_shape (NCHW)
///     4, 4, 2, 1, // input_strides
///     0,          // offset
///     1, 1, 4, 4, // output_shape
///     1,          // mode (linear/bilinear)
///     0,          // coord_transform (half_pixel)
///     0,          // nearest_mode (floor, not used for linear)
/// ];
/// call_ops_resize(&device, &command_buffer, &kernels, resize::F32,
///                 input_buffer, &output, &metadata)?;
/// ```
#[allow(clippy::too_many_arguments)]
pub fn call_ops_resize(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Resize, kernel.0)?;

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
