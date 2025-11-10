use crate::{
    compat::*,
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(
    reduce_window_max,
    reduce_window_min,
    reduce_window_sum,
    reduce_window_mean
);

/// Execute a reduce_window operation (sliding window reduction)
///
/// Performs sliding window reductions like max, min, sum, or mean.
/// Note: For production use, consider using cuDNN for optimal performance.
///
/// # Metadata layout
/// - metadata[0]: output_size (total number of elements in output)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: input_strides
/// - metadata[2+2*num_dims]: input_offset (starting offset in input)
/// - metadata[3+2*num_dims..3+3*num_dims]: window_shape (size of window in each dimension)
/// - metadata[3+3*num_dims..3+4*num_dims]: strides (step size in each dimension)
/// - metadata[3+4*num_dims..3+4*num_dims+2*num_dims]: padding (before and after for each dimension)
/// - metadata[3+6*num_dims..]: output_shape
///
/// # Kernel signature
/// `(input, output, metadata)`
///
/// # Algorithm
/// For each output element:
/// 1. Compute output coordinates
/// 2. For each position in the window:
///    - Compute corresponding input coordinates with stride and padding
///    - Check if position is within bounds (considering padding)
///    - Apply reduction operation on valid values
/// 3. Out-of-bounds values are treated according to operation:
///    - max: -infinity, min: +infinity, sum/mean: 0
pub fn call_ops_reduce_window<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    device: &Arc<CudaDevice>,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(device, Source::OpsWindowing, kernel.0)?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = ((num_els as u32 + block_size - 1) / block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let stream = device.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    // Kernel signature: (input, output, metadata)
    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}
