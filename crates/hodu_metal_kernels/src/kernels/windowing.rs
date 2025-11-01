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
    reduce_window_max,
    reduce_window_min,
    reduce_window_sum,
    reduce_window_mean
);

/// Executes a reduce_window operation (sliding window reduction) using Metal compute pipeline.
///
/// Performs reduction operations over sliding windows of the input tensor. This is commonly used
/// for pooling operations (max pooling, average pooling) in neural networks. The operation
/// slides a window across the input tensor and applies a reduction (max, min, sum, mean) to
/// each window, producing output elements.
///
/// # Arguments
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `kernels` - Kernel cache
/// * `kernel_name` - Reduce window kernel (reduce_window_max::F32, reduce_window_min::F32,
///   reduce_window_sum::F32, reduce_window_mean::F32)
/// * `input` - Input tensor buffer
/// * `output` - Output buffer for reduced windows
/// * `metadata` - Metadata describing tensor shapes, window parameters, strides, and padding
///
/// # Metadata Layout
/// Variable length based on tensor dimensionality:
///
/// - `metadata[0]`: output_size (num_els, total number of output elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: input_shape
/// - `metadata[2+num_dims..2+2*num_dims]`: input_strides
/// - `metadata[2+2*num_dims]`: input_offset
/// - `metadata[3+2*num_dims..3+3*num_dims]`: window_shape (size of sliding window in each dimension)
/// - `metadata[3+3*num_dims..3+4*num_dims]`: strides (step size for sliding window in each dimension)
/// - `metadata[3+4*num_dims..3+6*num_dims]`: padding (before and after padding for each dimension)
/// - `metadata[3+6*num_dims..]`: output_shape
///
/// Total metadata length: `3 + 6*num_dims + num_output_dims`
///
/// Note: Padding is stored as [pad_before_0, pad_after_0, pad_before_1, pad_after_1, ...]
///
/// # Supported Operations
/// - `reduce_window_max`: Maximum value in each window (max pooling)
/// - `reduce_window_min`: Minimum value in each window (min pooling)
/// - `reduce_window_sum`: Sum of values in each window
/// - `reduce_window_mean`: Mean (average) of values in each window (average pooling)
///
/// # Example - 1D Max Pooling
/// ```ignore
/// // Input: [1, 2, 3, 4, 5], Window: [2], Stride: [1]
/// // Output: [2, 3, 4, 5] (max of sliding windows)
/// let metadata = vec![
///     4,          // output_size
///     1,          // num_dims
///     5,          // input_shape
///     1,          // input_strides
///     0,          // input_offset
///     2,          // window_shape
///     1,          // stride
///     0, 0,       // padding (before, after)
///     4,          // output_shape
/// ];
/// call_reduce_window(&device, &command_buffer, &kernels, reduce_window_max::F32,
///                    input_buffer, &output, &metadata)?;
/// ```
///
/// # Example - 2D Max Pooling (2x2 with stride 2)
/// ```ignore
/// // Input: 4x4 matrix, Window: [2, 2], Stride: [2, 2]
/// // Output: 2x2 matrix (non-overlapping max pooling)
/// let metadata = vec![
///     4,          // output_size (2x2)
///     2,          // num_dims
///     4, 4,       // input_shape
///     4, 1,       // input_strides
///     0,          // input_offset
///     2, 2,       // window_shape
///     2, 2,       // strides
///     0, 0, 0, 0, // padding (h_before, h_after, w_before, w_after)
///     2, 2,       // output_shape
/// ];
/// call_reduce_window(&device, &command_buffer, &kernels, reduce_window_max::F32,
///                    input_buffer, &output, &metadata)?;
/// ```
pub fn call_reduce_window(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Windowing, kernel_name.0)?;

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
