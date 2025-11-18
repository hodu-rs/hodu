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
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsConv, kernel.0)?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

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
            args.arg(input).arg(weight).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute a convolution weight gradient operation for backpropagation
///
/// # Metadata layout (Generic, dimension-agnostic)
///
/// All grad_weight operations use a unified metadata structure:
///
/// - metadata[0]: num_els (total grad_weight elements)
/// - metadata[1]: input_ndim
/// - metadata[2]: spatial_dims
/// - metadata[3..3+input_ndim]: input_shape
/// - metadata[3+input_ndim..3+2*input_ndim]: grad_output_shape
/// - metadata[3+2*input_ndim..3+3*input_ndim]: weight_shape
/// - metadata[3+3*input_ndim..3+4*input_ndim]: input_strides
/// - metadata[3+4*input_ndim..3+5*input_ndim]: grad_output_strides
/// - metadata[3+5*input_ndim]: input_offset
/// - metadata[3+5*input_ndim+1]: grad_output_offset
/// - metadata[3+5*input_ndim+2..]: stride, padding, dilation (spatial_dims elements each)
///
/// ## Examples:
///
/// Conv1D (input_ndim=3, spatial_dims=1):
/// - metadata[18]: input_offset, metadata[19]: grad_output_offset
/// - metadata[20]: stride, metadata[21]: padding, metadata[22]: dilation
///
/// Conv2D (input_ndim=4, spatial_dims=2):
/// - metadata[23]: input_offset, metadata[24]: grad_output_offset
/// - metadata[25..27]: stride, metadata[27..29]: padding, metadata[29..31]: dilation
///
/// Conv3D (input_ndim=5, spatial_dims=3):
/// - metadata[28]: input_offset, metadata[29]: grad_output_offset
/// - metadata[30..33]: stride, metadata[33..36]: padding, metadata[36..39]: dilation
///
/// Transpose convolutions use the same layout.
pub fn call_ops_conv_grad_weight<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    grad_output: &CudaSlice<T>,
    grad_weight: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsConv, kernel.0)?;

    let num_els = metadata[0];
    let block_size = 256u32;
    let grid_size = (num_els as u32).div_ceil(block_size).max(1);

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
            args.arg(input).arg(grad_output).arg(grad_weight).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}
