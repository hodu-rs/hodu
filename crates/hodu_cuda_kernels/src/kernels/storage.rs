use crate::{
    compat::*,
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(const_set);

/// Executes a constant fill operation to set all elements of a tensor to a constant value.
///
/// This operation fills a tensor (possibly with strided layout) with a single constant value.
/// It supports non-contiguous layouts, so only the logical tensor elements are modified,
/// leaving gaps in strided buffers untouched.
///
/// # Arguments
/// * `kernel` - Const set kernel (e.g., const_set::F32)
/// * `device` - CUDA device to execute on
/// * `output` - Output device slice (will be filled with const_val)
/// * `metadata` - Device slice containing metadata describing tensor shape, strides, and offset
/// * `const_val` - Constant value to fill the tensor with
///
/// # Metadata Layout
/// Total metadata length: `2 + num_dims * 2 + 1`
///
/// - `metadata[0]`: num_els (total number of elements to set)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: shape (dimensions of the tensor)
/// - `metadata[2+num_dims..2+2*num_dims]`: strides (stride for each dimension)
/// - `metadata[2+2*num_dims]`: offset (starting offset in output buffer)
///
/// # Kernel signature
/// `(output, const_val, metadata)`
///
/// # Type Parameter
/// * `T: cudarc::driver::DeviceRepr` - The type of the constant value (f32, i32, bool, etc.)
///
/// # Example
/// ```ignore
/// // Fill a 3x4 matrix with value 7.0
/// let metadata = vec![
///     12,     // num_els (3 * 4)
///     2,      // num_dims
///     3, 4,   // shape
///     4, 1,   // strides (row-major)
///     0,      // offset
/// ];
/// call_const_set(const_set::F32, &device,
///                &output, &metadata, 7.0f32)?;
/// ```
///
/// # Safety
/// This function is unsafe because it launches a CUDA kernel. The caller must ensure:
/// - All device pointers are valid
/// - The metadata accurately describes the tensor layout
/// - The output buffer has sufficient capacity
pub fn call_const_set<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
    const_val: T,
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr + Clone,
{
    let func = kernels.load_function(device, Source::Storage, kernel.0)?;

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

    // Kernel signature: (output, const_val, metadata)
    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(output).arg(&const_val).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Synchronous version that automatically syncs the device after launch
pub fn call_const_set_sync<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    device: &Arc<CudaDevice>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
    const_val: T,
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr + Clone,
{
    call_const_set(kernel, kernels, device, output, metadata, const_val)?;
    device
        .synchronize()
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to synchronize: {:?}", e)))?;
    Ok(())
}
