use crate::{
    cuda::*,
    error::{CudaKernelError, Result},
    kernel::Kernels,
    kernels::macros::ops,
    source::Source,
};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use half::{bf16, f16};

ops!(matmul, dot);

macro_rules! impl_cublas_matmul {
    ($ty:ty) => {
        paste::paste! {
            #[doc = "cuBLAS-accelerated matmul for " $ty]
            #[allow(dead_code)]
            pub fn [<call_ops_matmul_cublas_ $ty>](
                _kernel: crate::kernels::macros::Kernel,
                _kernels: &Kernels,
                context: &Arc<CudaContext>,
                lhs: &CudaSlice<$ty>,
                rhs: &CudaSlice<$ty>,
                output: &mut CudaSlice<$ty>,
                metadata: &[usize],
            ) -> Result<()> {
                // Try cuBLAS first, fall back to kernel on error
                match call_ops_matmul_cublas(context, lhs, rhs, output, metadata) {
                    Ok(()) => Ok(()),
                    Err(_) => {
                        // Fall back to custom kernel
                        call_ops_matmul_kernel(_kernel, _kernels, context, lhs, rhs, output, metadata)
                    }
                }
            }
        }
    };
}

/// Helper trait to enable cuBLAS GEMM for supported types
trait CublasGemm: cudarc::driver::DeviceRepr + Sized {
    fn one() -> Self;
    fn zero() -> Self;

    fn cublas_gemm<A, B, C>(
        blas: &CudaBlas,
        cfg: GemmConfig<Self>,
        lhs: &A,
        rhs: &B,
        output: &mut C,
    ) -> core::result::Result<(), cudarc::cublas::result::CublasError>
    where
        A: cudarc::driver::DevicePtr<Self>,
        B: cudarc::driver::DevicePtr<Self>,
        C: cudarc::driver::DevicePtrMut<Self>;

    fn cublas_gemm_batched<A, B, C>(
        blas: &CudaBlas,
        cfg: StridedBatchedConfig<Self>,
        lhs: &A,
        rhs: &B,
        output: &mut C,
    ) -> core::result::Result<(), cudarc::cublas::result::CublasError>
    where
        A: cudarc::driver::DevicePtr<Self>,
        B: cudarc::driver::DevicePtr<Self>,
        C: cudarc::driver::DevicePtrMut<Self>;
}

impl CublasGemm for bf16 {
    fn one() -> Self {
        bf16::from_f32(1.0)
    }

    fn zero() -> Self {
        bf16::from_f32(0.0)
    }

    fn cublas_gemm<A, B, C>(
        blas: &CudaBlas,
        cfg: GemmConfig<Self>,
        lhs: &A,
        rhs: &B,
        output: &mut C,
    ) -> core::result::Result<(), cudarc::cublas::result::CublasError>
    where
        A: cudarc::driver::DevicePtr<Self>,
        B: cudarc::driver::DevicePtr<Self>,
        C: cudarc::driver::DevicePtrMut<Self>,
    {
        unsafe { blas.gemm(cfg, lhs, rhs, output) }
    }

    fn cublas_gemm_batched<A, B, C>(
        blas: &CudaBlas,
        cfg: StridedBatchedConfig<Self>,
        lhs: &A,
        rhs: &B,
        output: &mut C,
    ) -> core::result::Result<(), cudarc::cublas::result::CublasError>
    where
        A: cudarc::driver::DevicePtr<Self>,
        B: cudarc::driver::DevicePtr<Self>,
        C: cudarc::driver::DevicePtrMut<Self>,
    {
        unsafe { blas.gemm_strided_batched(cfg, lhs, rhs, output) }
    }
}

impl CublasGemm for f16 {
    fn one() -> Self {
        f16::from_f32(1.0)
    }

    fn zero() -> Self {
        f16::from_f32(0.0)
    }

    fn cublas_gemm<A, B, C>(
        blas: &CudaBlas,
        cfg: GemmConfig<Self>,
        lhs: &A,
        rhs: &B,
        output: &mut C,
    ) -> core::result::Result<(), cudarc::cublas::result::CublasError>
    where
        A: cudarc::driver::DevicePtr<Self>,
        B: cudarc::driver::DevicePtr<Self>,
        C: cudarc::driver::DevicePtrMut<Self>,
    {
        unsafe { blas.gemm(cfg, lhs, rhs, output) }
    }

    fn cublas_gemm_batched<A, B, C>(
        blas: &CudaBlas,
        cfg: StridedBatchedConfig<Self>,
        lhs: &A,
        rhs: &B,
        output: &mut C,
    ) -> core::result::Result<(), cudarc::cublas::result::CublasError>
    where
        A: cudarc::driver::DevicePtr<Self>,
        B: cudarc::driver::DevicePtr<Self>,
        C: cudarc::driver::DevicePtrMut<Self>,
    {
        unsafe { blas.gemm_strided_batched(cfg, lhs, rhs, output) }
    }
}

impl CublasGemm for f32 {
    fn one() -> Self {
        1.0
    }

    fn zero() -> Self {
        0.0
    }

    fn cublas_gemm<A, B, C>(
        blas: &CudaBlas,
        cfg: GemmConfig<Self>,
        lhs: &A,
        rhs: &B,
        output: &mut C,
    ) -> core::result::Result<(), cudarc::cublas::result::CublasError>
    where
        A: cudarc::driver::DevicePtr<Self>,
        B: cudarc::driver::DevicePtr<Self>,
        C: cudarc::driver::DevicePtrMut<Self>,
    {
        unsafe { blas.gemm(cfg, lhs, rhs, output) }
    }

    fn cublas_gemm_batched<A, B, C>(
        blas: &CudaBlas,
        cfg: StridedBatchedConfig<Self>,
        lhs: &A,
        rhs: &B,
        output: &mut C,
    ) -> core::result::Result<(), cudarc::cublas::result::CublasError>
    where
        A: cudarc::driver::DevicePtr<Self>,
        B: cudarc::driver::DevicePtr<Self>,
        C: cudarc::driver::DevicePtrMut<Self>,
    {
        unsafe { blas.gemm_strided_batched(cfg, lhs, rhs, output) }
    }
}

impl CublasGemm for f64 {
    fn one() -> Self {
        1.0
    }

    fn zero() -> Self {
        0.0
    }

    fn cublas_gemm<A, B, C>(
        blas: &CudaBlas,
        cfg: GemmConfig<Self>,
        lhs: &A,
        rhs: &B,
        output: &mut C,
    ) -> core::result::Result<(), cudarc::cublas::result::CublasError>
    where
        A: cudarc::driver::DevicePtr<Self>,
        B: cudarc::driver::DevicePtr<Self>,
        C: cudarc::driver::DevicePtrMut<Self>,
    {
        unsafe { blas.gemm(cfg, lhs, rhs, output) }
    }

    fn cublas_gemm_batched<A, B, C>(
        blas: &CudaBlas,
        cfg: StridedBatchedConfig<Self>,
        lhs: &A,
        rhs: &B,
        output: &mut C,
    ) -> core::result::Result<(), cudarc::cublas::result::CublasError>
    where
        A: cudarc::driver::DevicePtr<Self>,
        B: cudarc::driver::DevicePtr<Self>,
        C: cudarc::driver::DevicePtrMut<Self>,
    {
        unsafe { blas.gemm_strided_batched(cfg, lhs, rhs, output) }
    }
}

/// Execute matmul using cuBLAS (for bf16/f16/f32/f64)
fn call_ops_matmul_cublas<T>(
    context: &Arc<CudaContext>,
    lhs: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: CublasGemm,
{
    let lhs_ndim = metadata[1];
    let rhs_ndim = metadata[2];
    let batch_ndim = metadata[3];

    let metadata_base = 4 + lhs_ndim + rhs_ndim + batch_ndim + lhs_ndim + rhs_ndim;
    let lhs_offset = metadata[metadata_base];
    let rhs_offset = metadata[metadata_base + 1];
    let m = metadata[metadata_base + 2];
    let k = metadata[metadata_base + 3];
    let n = metadata[metadata_base + 4];

    // Calculate batch count
    let num_batches = if batch_ndim == 0 {
        1
    } else {
        let batch_shape = &metadata[4 + lhs_ndim + rhs_ndim..4 + lhs_ndim + rhs_ndim + batch_ndim];
        batch_shape.iter().product()
    };

    // Extract strides from metadata
    let lhs_strides = &metadata[4 + lhs_ndim + rhs_ndim + batch_ndim..4 + 2 * lhs_ndim + rhs_ndim + batch_ndim];
    let rhs_strides = &metadata[4 + 2 * lhs_ndim + rhs_ndim + batch_ndim..4 + 2 * lhs_ndim + 2 * rhs_ndim + batch_ndim];

    // Check if last dimension is contiguous (stride == 1 for row-major)
    // cuBLAS can handle non-contiguous matrices via lda/ldb parameters
    let lhs_last_contiguous = lhs_strides[lhs_ndim - 1] == 1;
    let rhs_last_contiguous = rhs_strides[rhs_ndim - 1] == 1;

    if lhs_last_contiguous && rhs_last_contiguous {
        // Use cuBLAS with proper leading dimensions
        let stream = context.default_stream();
        let blas = CudaBlas::new(stream)
            .map_err(|e| CudaKernelError::LaunchError(format!("Failed to create cuBLAS: {:?}", e)))?;

        // Leading dimensions (strides of second-to-last dimension)
        let mut lda = lhs_strides[lhs_ndim - 2] as i32;
        let mut ldb = rhs_strides[rhs_ndim - 2] as i32;

        // Fix leading dimensions when they're too small
        // For row-major contiguous matrices: lda >= K, ldb >= N
        if lda < k as i32 {
            lda = k as i32;
        }
        if ldb < n as i32 {
            ldb = n as i32;
        }

        // Batch strides (stride of first dimension if batch exists)
        let lhs_batch_stride = if batch_ndim > 0 { lhs_strides[0] as i64 } else { 0 };
        let rhs_batch_stride = if batch_ndim > 0 { rhs_strides[0] as i64 } else { 0 };

        // Offset slices
        let lhs_view = lhs.slice(lhs_offset..);
        let rhs_view = rhs.slice(rhs_offset..);

        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                m: n as i32, // cuBLAS uses column-major, so swap m and n
                n: m as i32,
                k: k as i32,
                alpha: T::one(),
                lda: ldb, // Use rhs stride for first operand (swapped)
                ldb: lda, // Use lhs stride for second operand (swapped)
                beta: T::zero(),
                ldc: n as i32,
            },
            batch_size: num_batches as i32,
            stride_a: rhs_batch_stride, // rhs batch stride (swapped)
            stride_b: lhs_batch_stride, // lhs batch stride (swapped)
            stride_c: (m * n) as i64,
        };

        // Note: cuBLAS expects column-major, so we swap lhs and rhs
        T::cublas_gemm_batched(&blas, cfg, &rhs_view, &lhs_view, output)
            .map_err(|e| CudaKernelError::LaunchError(format!("cuBLAS GEMM failed: {:?}", e)))?;

        Ok(())
    } else {
        Err(CudaKernelError::LaunchError(
            "Only row-major matrices with contiguous last dimension are supported by cuBLAS path".into(),
        ))
    }
}

/// Execute a batched matrix multiplication with broadcasting support (generic kernel-based version)
fn call_ops_matmul_kernel<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    lhs: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsMatrix, kernel.0)?;

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

    // For matrix multiplication, we use 2D thread blocks with tiling
    const TILE_SIZE: u32 = 16;

    let grid_width = (n as u32).div_ceil(TILE_SIZE).max(1);
    let grid_height = (m as u32).div_ceil(TILE_SIZE).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_width, grid_height, num_batches as u32),
        block_dim: (TILE_SIZE, TILE_SIZE, 1),
        shared_mem_bytes: 0,
    };

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(lhs).arg(rhs).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute a batched matrix multiplication with broadcasting support
///
/// This function attempts to use cuBLAS for bf16/f16/f32/f64 types when matrices are contiguous,
/// falling back to custom CUDA kernels for non-contiguous cases or other types.
///
/// # Arguments
/// * `kernel` - The matmul kernel (e.g., "matmul::F32")
/// * `kernels` - Kernel cache
/// * `context` - CUDA context to execute on
/// * `lhs` - Left-hand side matrix device slice with shape [..., M, K]
/// * `rhs` - Right-hand side matrix device slice with shape [..., K, N]
/// * `output` - Output matrix device slice with shape [..., M, N]
/// * `metadata` - Device slice containing metadata describing matrix dimensions
///
/// # Metadata layout
/// - metadata[0]: num_els (total output elements)
/// - metadata[1]: lhs_ndim (number of dimensions in lhs)
/// - metadata[2]: rhs_ndim (number of dimensions in rhs)
/// - metadata[3]: batch_ndim (number of batch dimensions)
/// - metadata[4..4+lhs_ndim]: lhs_shape
/// - metadata[4+lhs_ndim..4+lhs_ndim+rhs_ndim]: rhs_shape
/// - metadata[4+lhs_ndim+rhs_ndim..4+lhs_ndim+rhs_ndim+batch_ndim]: batch_shape
/// - metadata[4+lhs_ndim+rhs_ndim+batch_ndim..4+2*lhs_ndim+rhs_ndim+batch_ndim]: lhs_strides
/// - metadata[4+2*lhs_ndim+rhs_ndim+batch_ndim..4+2*lhs_ndim+2*rhs_ndim+batch_ndim]: rhs_strides
/// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim]: lhs_offset
/// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+1]: rhs_offset
/// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+2]: M (rows of A)
/// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+3]: K (cols of A / rows of B)
/// - metadata[4+2*lhs_ndim+2*rhs_ndim+batch_ndim+4]: N (cols of B)
pub fn call_ops_matmul<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    lhs: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    // Use custom kernel (cuBLAS integration will be added via specialized implementations)
    call_ops_matmul_kernel(kernel, kernels, context, lhs, rhs, output, metadata)
}

// Specialized implementations for bf16, f16, f32, and f64 that use cuBLAS
impl_cublas_matmul!(bf16);
impl_cublas_matmul!(f16);
impl_cublas_matmul!(f32);
impl_cublas_matmul!(f64);

macro_rules! impl_cublas_dot {
    ($ty:ty) => {
        paste::paste! {
            #[doc = "cuBLAS-accelerated dot for " $ty]
            #[allow(dead_code)]
            pub fn [<call_ops_dot_cublas_ $ty>](
                _kernel: crate::kernels::macros::Kernel,
                _kernels: &Kernels,
                context: &Arc<CudaContext>,
                lhs: &CudaSlice<$ty>,
                rhs: &CudaSlice<$ty>,
                output: &mut CudaSlice<$ty>,
                metadata: &[usize],
            ) -> Result<()> {
                // Try cuBLAS first, fall back to kernel on error
                match call_ops_dot_cublas(context, lhs, rhs, output, metadata) {
                    Ok(()) => Ok(()),
                    Err(_) => {
                        // Fall back to custom kernel
                        call_ops_dot_kernel(_kernel, _kernels, context, lhs, rhs, output, metadata)
                    }
                }
            }
        }
    };
}

/// Execute dot using cuBLAS (for bf16/f16/f32/f64)
fn call_ops_dot_cublas<T>(
    context: &Arc<CudaContext>,
    lhs: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: CublasGemm,
{
    let m = metadata[0];
    let k = metadata[1];
    let n = metadata[2];
    let lhs_stride_m = metadata[3];
    let lhs_stride_k = metadata[4];
    let rhs_stride_k = metadata[5];
    let rhs_stride_n = metadata[6];
    let lhs_offset = metadata[7];
    let rhs_offset = metadata[8];

    // Check if last dimension is contiguous (stride == 1 for row-major)
    // cuBLAS can handle non-contiguous matrices via lda/ldb parameters
    let lhs_last_contiguous = lhs_stride_k == 1;
    let rhs_last_contiguous = rhs_stride_n == 1;

    if lhs_last_contiguous && rhs_last_contiguous {
        // Use cuBLAS with proper leading dimensions
        let stream = context.default_stream();
        let blas = CudaBlas::new(stream)
            .map_err(|e| CudaKernelError::LaunchError(format!("Failed to create cuBLAS: {:?}", e)))?;

        // Leading dimensions (strides of first dimension for 2D matrices)
        let mut lda = lhs_stride_m as i32;
        let mut ldb = rhs_stride_k as i32;

        // Fix leading dimensions when they're too small
        // For row-major contiguous matrices: lda >= K, ldb >= N
        if lda < k as i32 {
            lda = k as i32;
        }
        if ldb < n as i32 {
            ldb = n as i32;
        }

        // Offset slices
        let lhs_view = lhs.slice(lhs_offset..);
        let rhs_view = rhs.slice(rhs_offset..);

        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32, // cuBLAS uses column-major, so swap m and n
            n: m as i32,
            k: k as i32,
            alpha: T::one(),
            lda: ldb, // Use rhs stride for first operand (swapped)
            ldb: lda, // Use lhs stride for second operand (swapped)
            beta: T::zero(),
            ldc: n as i32,
        };

        // Note: cuBLAS expects column-major, so we swap lhs and rhs
        T::cublas_gemm(&blas, cfg, &rhs_view, &lhs_view, output)
            .map_err(|e| CudaKernelError::LaunchError(format!("cuBLAS GEMM failed: {:?}", e)))?;

        Ok(())
    } else {
        Err(CudaKernelError::LaunchError(
            "Only row-major matrices with contiguous last dimension are supported by cuBLAS path".into(),
        ))
    }
}

/// Execute a 2D matrix multiplication using custom CUDA kernel (generic kernel-based version)
fn call_ops_dot_kernel<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    lhs: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    let func = kernels.load_function(context, Source::OpsMatrix, kernel.0)?;

    // Extract matrix dimensions from metadata
    let m = metadata[0];
    let n = metadata[2];

    // Optimized dot product with register blocking (4x4 per thread)
    // Tile size is 32x32, with 8x8 threadgroups
    const DOT_TILE_SIZE: u32 = 32;
    const BLOCK_SIZE: u32 = 4;
    const THREADS_PER_DIM: u32 = DOT_TILE_SIZE / BLOCK_SIZE; // 8

    let grid_width = (n as u32).div_ceil(DOT_TILE_SIZE).max(1);
    let grid_height = (m as u32).div_ceil(DOT_TILE_SIZE).max(1);

    let cfg = LaunchConfig {
        grid_dim: (grid_width, grid_height, 1),
        block_dim: (THREADS_PER_DIM, THREADS_PER_DIM, 1),
        shared_mem_bytes: 0,
    };

    let stream = context.default_stream();
    let metadata_dev = stream
        .memcpy_stod(metadata)
        .map_err(|e| CudaKernelError::MemoryError(format!("Failed to copy metadata: {:?}", e)))?;

    unsafe {
        func.launch(&stream, cfg, |args| {
            args.arg(lhs).arg(rhs).arg(output).arg(&metadata_dev);
        })
        .map_err(|e| CudaKernelError::LaunchError(format!("Failed to launch kernel: {:?}", e)))?;
    }

    Ok(())
}

/// Execute a 2D matrix multiplication
///
/// This function attempts to use cuBLAS for bf16/f16/f32/f64 types when matrices are contiguous,
/// falling back to custom CUDA kernels for non-contiguous cases or other types.
///
/// # Arguments
/// * `kernel` - The dot kernel (e.g., "dot::F32")
/// * `kernels` - Kernel cache
/// * `context` - CUDA context to execute on
/// * `lhs` - Left input matrix device slice with shape [M, K]
/// * `rhs` - Right input matrix device slice with shape [K, N]
/// * `output` - Output matrix device slice with shape [M, N]
/// * `metadata` - Device slice containing metadata describing matrix layout
///
/// # Metadata layout
/// - metadata[0]: M (rows of A)
/// - metadata[1]: K (cols of A / rows of B)
/// - metadata[2]: N (cols of B)
/// - metadata[3]: lhs_stride_m (stride for row dimension of A)
/// - metadata[4]: lhs_stride_k (stride for col dimension of A)
/// - metadata[5]: rhs_stride_k (stride for row dimension of B)
/// - metadata[6]: rhs_stride_n (stride for col dimension of B)
/// - metadata[7]: lhs_offset (starting offset in lhs buffer)
/// - metadata[8]: rhs_offset (starting offset in rhs buffer)
pub fn call_ops_dot<T>(
    kernel: crate::kernels::macros::Kernel,
    kernels: &Kernels,
    context: &Arc<CudaContext>,
    lhs: &CudaSlice<T>,
    rhs: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    metadata: &[usize],
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr,
{
    // Use custom kernel (cuBLAS integration will be added via specialized implementations)
    call_ops_dot_kernel(kernel, kernels, context, lhs, rhs, output, metadata)
}

// Specialized implementations for bf16, f16, f32, and f64 that use cuBLAS
impl_cublas_dot!(bf16);
impl_cublas_dot!(f16);
impl_cublas_dot!(f32);
impl_cublas_dot!(f64);
