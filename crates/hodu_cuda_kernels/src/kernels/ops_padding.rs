//! Padding operations
//!
//! This module provides padding operations for tensors:
//! - pad_constant: Pad with a constant value
//! - pad_reflect: Pad with reflected values at boundaries
//! - pad_replicate: Pad by replicating edge values
//! - pad_circular: Pad with circular/wrapped values

use crate::{
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};

ops!(pad_constant, pad_reflect, pad_replicate, pad_circular);

/// Execute a constant padding operation
///
/// Pads tensor with a constant value.
///
/// # Arguments
/// * `kernel` - The padding kernel (e.g., pad_constant::F32)
/// * `kernels` - Kernel cache
/// * `context` - CUDA context
/// * `input` - Input tensor device slice
/// * `output` - Output tensor device slice (padded result)
/// * `pad_value` - Device slice containing the constant pad value
/// * `metadata` - Metadata describing tensor shapes and padding
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: output_shape
/// - metadata[2+2*num_dims..2+3*num_dims]: pad_before (padding before each dim)
pub fn call_ops_pad_constant<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    pad_value: &CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsPadding, kernel.0)?;

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
            args.arg(input).arg(output).arg(pad_value).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute a reflect padding operation
///
/// Pads tensor with reflected values at boundaries.
/// For input [1, 2, 3] with pad=2: [3, 2, 1, 2, 3, 2, 1]
///
/// # Arguments
/// * `kernel` - The padding kernel (e.g., pad_reflect::F32)
/// * `kernels` - Kernel cache
/// * `context` - CUDA context
/// * `input` - Input tensor device slice
/// * `output` - Output tensor device slice (padded result)
/// * `metadata` - Metadata describing tensor shapes and padding
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: output_shape
/// - metadata[2+2*num_dims..2+3*num_dims]: pad_before (padding before each dim)
pub fn call_ops_pad_reflect<T>(
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
    let func = kernels.load_function(context, Source::OpsPadding, kernel.0)?;

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
            args.arg(input).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute a replicate (edge) padding operation
///
/// Pads tensor by replicating edge values.
/// For input [1, 2, 3] with pad=2: [1, 1, 1, 2, 3, 3, 3]
///
/// # Arguments
/// * `kernel` - The padding kernel (e.g., pad_replicate::F32)
/// * `kernels` - Kernel cache
/// * `context` - CUDA context
/// * `input` - Input tensor device slice
/// * `output` - Output tensor device slice (padded result)
/// * `metadata` - Metadata describing tensor shapes and padding
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: output_shape
/// - metadata[2+2*num_dims..2+3*num_dims]: pad_before (padding before each dim)
pub fn call_ops_pad_replicate<T>(
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
    let func = kernels.load_function(context, Source::OpsPadding, kernel.0)?;

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
            args.arg(input).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute a circular (wrap) padding operation
///
/// Pads tensor with circular/wrapped values.
/// For input [1, 2, 3] with pad=2: [2, 3, 1, 2, 3, 1, 2]
///
/// # Arguments
/// * `kernel` - The padding kernel (e.g., pad_circular::F32)
/// * `kernels` - Kernel cache
/// * `context` - CUDA context
/// * `input` - Input tensor device slice
/// * `output` - Output tensor device slice (padded result)
/// * `metadata` - Metadata describing tensor shapes and padding
///
/// # Metadata layout
/// - metadata[0]: num_els (total number of output elements)
/// - metadata[1]: num_dims (number of dimensions)
/// - metadata[2..2+num_dims]: input_shape
/// - metadata[2+num_dims..2+2*num_dims]: output_shape
/// - metadata[2+2*num_dims..2+3*num_dims]: pad_before (padding before each dim)
pub fn call_ops_pad_circular<T>(
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
    let func = kernels.load_function(context, Source::OpsPadding, kernel.0)?;

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
            args.arg(input).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}
