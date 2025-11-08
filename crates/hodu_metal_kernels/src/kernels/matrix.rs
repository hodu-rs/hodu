use crate::{
    error::MetalKernelError,
    kernel::Kernels,
    kernels::macros::ops,
    metal::{Buffer, ComputeCommandEncoder, Device},
    set_params,
    source::Source,
    utils::{BufferOffset, EncoderProvider},
};
use objc2_metal::{MTLResourceUsage, MTLSize};

ops!(matmul, dot);

/// Executes a batched matrix multiplication with broadcasting support using Metal compute pipeline.
///
/// Performs matrix multiplication C = A @ B with support for batch dimensions and broadcasting.
/// The last two dimensions are treated as the matrix dimensions (M×K) @ (K×N) = (M×N),
/// while leading dimensions represent batch dimensions.
///
/// # Arguments
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `kernels` - Kernel cache
/// * `kernel_name` - Matmul kernel (e.g., matmul::F32)
/// * `lhs` - Left input tensor (A) with shape [..., M, K]
/// * `rhs` - Right input tensor (B) with shape [..., K, N]
/// * `output` - Output buffer for result with shape [..., M, N]
/// * `metadata` - Metadata describing matrix dimensions and layout
///
/// # Metadata Layout
/// Variable length based on tensor dimensionality:
///
/// - `metadata[0]`: num_els (total output elements)
/// - `metadata[1]`: lhs_ndim (number of dimensions in lhs)
/// - `metadata[2]`: rhs_ndim (number of dimensions in rhs)
/// - `metadata[3]`: batch_ndim (number of batch dimensions)
/// - `metadata[4..4+lhs_ndim]`: lhs_shape
/// - `metadata[4+lhs_ndim..4+lhs_ndim+rhs_ndim]`: rhs_shape
/// - `metadata[4+lhs_ndim+rhs_ndim..4+lhs_ndim+rhs_ndim+batch_ndim]`: batch_shape (broadcasted batch dimensions)
/// - `metadata[4+lhs_ndim+rhs_ndim+batch_ndim..4+2*lhs_ndim+rhs_ndim+batch_ndim]`: lhs_strides
/// - `metadata[4+2*lhs_ndim+rhs_ndim+batch_ndim..4+2*lhs_ndim+2*rhs_ndim+batch_ndim]`: rhs_strides
/// - `metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim]`: lhs_offset
/// - `metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+1]`: rhs_offset
/// - `metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+2]`: M (rows of A)
/// - `metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+3]`: K (cols of A / rows of B)
/// - `metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+4]`: N (cols of B)
///
/// # Example
/// ```ignore
/// // 2D matmul: [2,3] @ [3,2] = [2,2]
/// let metadata = vec![
///     4,      // num_els
///     2,      // lhs_ndim
///     2,      // rhs_ndim
///     0,      // batch_ndim
///     2, 3,   // lhs_shape
///     3, 2,   // rhs_shape
///     3, 1,   // lhs_strides
///     2, 1,   // rhs_strides
///     0, 0,   // offsets
///     2, 3, 2 // M, K, N
/// ];
/// call_matmul(&device, &command_buffer, &kernels, matmul::F32,
///             lhs_buffer, rhs_buffer, &output, &metadata)?;
/// ```
#[allow(clippy::too_many_arguments)]
pub fn call_matmul(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    lhs: BufferOffset,
    rhs: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Matrix, kernel_name.0)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(encoder, (&lhs, &rhs, output, metadata));

    encoder.use_resource(lhs.buffer, MTLResourceUsage::Read);
    encoder.use_resource(rhs.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    // Extract M, N, and batch info from metadata
    let lhs_ndim = metadata[1];
    let rhs_ndim = metadata[2];
    let batch_ndim = metadata[3];

    let metadata_base = 4 + lhs_ndim + rhs_ndim + batch_ndim + lhs_ndim + rhs_ndim;
    let m = metadata[metadata_base + 2];
    let n = metadata[metadata_base + 4];

    // Calculate total number of batches
    let num_batches = if batch_ndim == 0 {
        1
    } else {
        let batch_shape = &metadata[4 + lhs_ndim + rhs_ndim..4 + lhs_ndim + rhs_ndim + batch_ndim];
        batch_shape.iter().product()
    };

    // Use 3D tiled dispatch (similar to DOT kernel but with batch dimension)
    const TILE_SIZE: usize = 16;
    let threadgroup_size = MTLSize {
        width: TILE_SIZE,
        height: TILE_SIZE,
        depth: 1,
    };

    let grid_width = (n + TILE_SIZE - 1).div_ceil(TILE_SIZE);
    let grid_height = (m + TILE_SIZE - 1).div_ceil(TILE_SIZE);

    let threadgroup_count = MTLSize {
        width: grid_width,
        height: grid_height,
        depth: num_batches,
    };

    encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);

    Ok(())
}

/// Executes a tiled 2D matrix multiplication optimized with threadgroup (shared) memory.
///
/// Performs matrix multiplication C = A @ B using a tiled algorithm with 16×16 tiles.
/// This is optimized for 2D matrix multiplication (no batching) and uses shared memory
/// to reduce global memory access and improve performance.
///
/// # Arguments
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
/// * `kernels` - Kernel cache
/// * `kernel_name` - Dot kernel (e.g., dot::F32)
/// * `lhs` - Left input matrix (A) with shape [M, K]
/// * `rhs` - Right input matrix (B) with shape [K, N]
/// * `output` - Output buffer for result with shape [M, N]
/// * `m` - Number of rows in A (and output)
/// * `n` - Number of columns in B (and output)
/// * `metadata` - Metadata describing matrix layout
///
/// # Metadata Layout
/// Fixed length: 9 elements
///
/// - `metadata[0]`: M (rows of A)
/// - `metadata[1]`: K (cols of A / rows of B)
/// - `metadata[2]`: N (cols of B)
/// - `metadata[3]`: lhs_stride_m (stride for row dimension of A)
/// - `metadata[4]`: lhs_stride_k (stride for col dimension of A)
/// - `metadata[5]`: rhs_stride_k (stride for row dimension of B)
/// - `metadata[6]`: rhs_stride_n (stride for col dimension of B)
/// - `metadata[7]`: lhs_offset (starting offset in lhs buffer)
/// - `metadata[8]`: rhs_offset (starting offset in rhs buffer)
///
/// # Performance
/// Uses 16×16 tiles with threadgroup memory for better cache locality.
/// Best suited for moderately-sized 2D matrices. For very large matrices
/// or batched operations, consider using `call_matmul` instead.
///
/// # Example
/// ```ignore
/// // 2D dot product: [2,3] @ [3,2] = [2,2]
/// let metadata = vec![
///     2,  // M
///     3,  // K
///     2,  // N
///     3,  // lhs_stride_m
///     1,  // lhs_stride_k
///     2,  // rhs_stride_k
///     1,  // rhs_stride_n
///     0,  // lhs_offset
///     0,  // rhs_offset
/// ];
/// call_dot(&device, &command_buffer, &kernels, dot::F32,
///          lhs_buffer, rhs_buffer, &output, 2, 2, &metadata)?;
/// ```
#[allow(clippy::too_many_arguments)]
pub fn call_dot(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: Kernel,
    lhs: BufferOffset,
    rhs: BufferOffset,
    output: &Buffer,
    m: usize,
    n: usize,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Matrix, kernel_name.0)?;

    let num_els = m * n;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(encoder, (&lhs, &rhs, output, num_els, metadata));

    encoder.use_resource(lhs.buffer, MTLResourceUsage::Read);
    encoder.use_resource(rhs.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    // Optimized dot product with register blocking (4x4 per thread)
    // Tile size is 32x32, with 8x8 threadgroups (each thread handles 4x4 block)
    const DOT_TILE_SIZE: usize = 32;
    const BLOCK_SIZE: usize = 4;
    const THREADS_PER_DIM: usize = DOT_TILE_SIZE / BLOCK_SIZE; // 8

    let threadgroup_size = MTLSize {
        width: THREADS_PER_DIM,
        height: THREADS_PER_DIM,
        depth: 1,
    };

    let grid_width = (n + DOT_TILE_SIZE - 1).div_ceil(DOT_TILE_SIZE);
    let grid_height = (m + DOT_TILE_SIZE - 1).div_ceil(DOT_TILE_SIZE);

    let threadgroup_count = MTLSize {
        width: grid_width,
        height: grid_height,
        depth: 1,
    };

    encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);

    Ok(())
}
