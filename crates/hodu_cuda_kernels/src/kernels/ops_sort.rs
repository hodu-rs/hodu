//! Sort operations
//!
//! This module provides sorting operations:
//! - topk: Get top-k largest or smallest elements along a dimension

use crate::{
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(topk);

/// Execute a topk operation
///
/// Returns the k largest or smallest elements along the last dimension.
///
/// # Arguments
/// * `kernel` - The topk kernel (e.g., topk::F32)
/// * `kernels` - Kernel cache
/// * `context` - CUDA context
/// * `input` - Input tensor device slice
/// * `values` - Output values device slice
/// * `indices` - Output indices device slice (i32)
/// * `metadata` - Metadata describing the operation
///
/// # Metadata layout
/// - metadata[0]: output_size (k * outer_size)
/// - metadata[1]: k (number of top elements)
/// - metadata[2]: last_dim_size (size of the dimension to search along)
/// - metadata[3]: outer_size (product of all dimensions except last)
/// - metadata[4]: largest (1 = largest, 0 = smallest)
/// - metadata[5]: sorted (1 = sorted, 0 = unsorted)
/// - metadata[6]: offset
pub fn call_topk<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    values: &mut CudaSlice<T>,
    indices: &mut CudaSlice<i32>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsSort, kernel.0)?;

    let outer_size = metadata[3];

    let block_size = 256u32;
    let grid_size = (outer_size as u32).div_ceil(block_size).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(values).arg(indices).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}
