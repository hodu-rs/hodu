use crate::{
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(resize);

/// Execute a resize operation (spatial interpolation)
///
/// Resizes the spatial dimensions of the input tensor using various interpolation modes.
/// Supports ONNX-compatible coordinate transformation modes.
///
/// # Metadata layout
/// - metadata[0]: output_size (total number of elements in output)
/// - metadata[1]: num_dims (number of dimensions, typically 4 for NCHW or 5 for NCDHW)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: input_strides
/// - metadata[2+2*num_dims]: offset (starting offset in input)
/// - metadata[3+2*num_dims..3+3*num_dims]: output_shape
/// - metadata[3+3*num_dims]: mode (0=nearest, 1=linear, 2=cubic)
/// - metadata[4+3*num_dims]: coord_transform (0=half_pixel, 1=asymmetric, 2=align_corners, 3=pytorch_half_pixel)
/// - metadata[5+3*num_dims]: nearest_mode (0=floor, 1=ceil, 2=round_prefer_floor, 3=round_prefer_ceil)
///
/// # Kernel signature
/// `(input, output, metadata)`
pub fn call_ops_resize<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsResize, kernel.0)?;

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

    // Kernel signature: (input, output, metadata)
    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(input).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}
