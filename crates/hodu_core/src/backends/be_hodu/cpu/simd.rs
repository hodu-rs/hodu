#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
pub mod neon;

pub mod ops;

pub trait SimdType: Copy + Default {
    type Vector: Copy;
    const LANES: usize;

    unsafe fn simd_load(ptr: *const Self) -> Self::Vector;
    #[allow(dead_code)]
    unsafe fn simd_store(ptr: *mut Self, vec: Self::Vector);
    unsafe fn simd_splat(val: Self) -> Self::Vector;
    unsafe fn simd_mul(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    unsafe fn simd_add(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    unsafe fn simd_fma(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector;
    unsafe fn simd_zero() -> Self::Vector;
    unsafe fn simd_horizontal_sum(vec: Self::Vector) -> Self;
}
