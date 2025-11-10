use crate::{
    compat::*,
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};

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

/// Execute a convolution operation (1D, 2D, 3D, or transposed variants)
/// Note: For production use, consider using cuDNN for optimal performance
pub fn call_ops_conv<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    device: &Arc<CudaDevice>,
    input: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(device, Source::OpsConv, kernel.0)?;

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

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(weight).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute a convolution weight gradient operation for backpropagation
pub fn call_ops_conv_grad_weight<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    device: &Arc<CudaDevice>,
    input: &CudaSlice<T>,
    grad_output: &CudaSlice<T>,
    grad_weight: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(device, Source::OpsConv, kernel.0)?;

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

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(grad_output).arg(grad_weight).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}
