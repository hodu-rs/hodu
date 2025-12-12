//! Einsum operations
//!
//! This module provides Einstein summation operations for tensor contractions.

use crate::{
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(einsum);

/// Execute an einsum operation
///
/// # Arguments
/// * `kernel` - The einsum kernel (e.g., einsum::F32)
/// * `kernels` - Kernel cache
/// * `context` - CUDA context
/// * `inputs` - Input tensor device slices (up to 4)
/// * `output` - Output tensor device slice
/// * `metadata` - Metadata describing einsum specification
pub fn call_ops_einsum<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    inputs: &[&CudaSlice<T>],
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsEinsum, kernel.0)?;

    let num_output_els = metadata[0];

    let block_size = 256u32;
    let grid_size = (num_output_els as u32).div_ceil(block_size).max(1);

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
            // Set input buffers (use first input for unused slots)
            for i in 0..4 {
                if i < inputs.len() {
                    args.arg(inputs[i]);
                } else {
                    args.arg(inputs[0]);
                }
            }
            args.arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}
