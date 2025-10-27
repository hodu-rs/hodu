use crate::backends::be_hodu::cpu::simd::SimdType;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use core::arch::arm::*;

impl SimdType for f32 {
    type Vector = float32x4_t;
    const LANES: usize = 4;

    #[inline]
    unsafe fn simd_load(ptr: *const Self) -> Self::Vector {
        vld1q_f32(ptr)
    }

    #[inline]
    unsafe fn simd_store(ptr: *mut Self, vec: Self::Vector) {
        vst1q_f32(ptr, vec);
    }

    #[inline]
    unsafe fn simd_splat(val: Self) -> Self::Vector {
        vdupq_n_f32(val)
    }

    #[inline]
    unsafe fn simd_mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        vmulq_f32(a, b)
    }

    #[inline]
    unsafe fn simd_add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        vaddq_f32(a, b)
    }

    #[inline]
    unsafe fn simd_fma(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        vfmaq_f32(c, a, b)
    }

    #[inline]
    unsafe fn simd_zero() -> Self::Vector {
        vdupq_n_f32(0.0)
    }

    #[inline]
    unsafe fn simd_horizontal_sum(vec: Self::Vector) -> Self {
        vaddvq_f32(vec)
    }
}

impl SimdType for f64 {
    type Vector = float64x2_t;
    const LANES: usize = 2;

    #[inline]
    unsafe fn simd_load(ptr: *const Self) -> Self::Vector {
        vld1q_f64(ptr)
    }

    #[inline]
    unsafe fn simd_store(ptr: *mut Self, vec: Self::Vector) {
        vst1q_f64(ptr, vec);
    }

    #[inline]
    unsafe fn simd_splat(val: Self) -> Self::Vector {
        vdupq_n_f64(val)
    }

    #[inline]
    unsafe fn simd_mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        vmulq_f64(a, b)
    }

    #[inline]
    unsafe fn simd_add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        vaddq_f64(a, b)
    }

    #[inline]
    unsafe fn simd_fma(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        vfmaq_f64(c, a, b)
    }

    #[inline]
    unsafe fn simd_zero() -> Self::Vector {
        vdupq_n_f64(0.0)
    }

    #[inline]
    unsafe fn simd_horizontal_sum(vec: Self::Vector) -> Self {
        vaddvq_f64(vec)
    }
}

macro_rules! impl_neon_int {
    ($ty:ty, $vec:ty, $lanes:expr, $dup:ident, $add:ident, $mul:ident, $addv:ident) => {
        impl SimdType for $ty {
            type Vector = $vec;
            const LANES: usize = $lanes;

            #[inline]
            unsafe fn simd_load(ptr: *const Self) -> Self::Vector {
                core::ptr::read_unaligned(ptr as *const $vec)
            }

            #[inline]
            unsafe fn simd_store(ptr: *mut Self, vec: Self::Vector) {
                core::ptr::write_unaligned(ptr as *mut $vec, vec);
            }

            #[inline]
            unsafe fn simd_splat(val: Self) -> Self::Vector {
                $dup(val)
            }

            #[inline]
            unsafe fn simd_mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
                $mul(a, b)
            }

            #[inline]
            unsafe fn simd_add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
                $add(a, b)
            }

            #[inline]
            unsafe fn simd_fma(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
                $add($mul(a, b), c)
            }

            #[inline]
            unsafe fn simd_zero() -> Self::Vector {
                $dup(0)
            }

            #[inline]
            unsafe fn simd_horizontal_sum(vec: Self::Vector) -> Self {
                $addv(vec)
            }
        }
    };
}

impl_neon_int!(u8, uint8x16_t, 16, vdupq_n_u8, vaddq_u8, vmulq_u8, vaddvq_u8);
impl_neon_int!(u16, uint16x8_t, 8, vdupq_n_u16, vaddq_u16, vmulq_u16, vaddvq_u16);
impl_neon_int!(u32, uint32x4_t, 4, vdupq_n_u32, vaddq_u32, vmulq_u32, vaddvq_u32);
impl_neon_int!(i8, int8x16_t, 16, vdupq_n_s8, vaddq_s8, vmulq_s8, vaddvq_s8);
impl_neon_int!(i16, int16x8_t, 8, vdupq_n_s16, vaddq_s16, vmulq_s16, vaddvq_s16);
impl_neon_int!(i32, int32x4_t, 4, vdupq_n_s32, vaddq_s32, vmulq_s32, vaddvq_s32);

impl SimdType for i64 {
    type Vector = int64x2_t;
    const LANES: usize = 2;

    #[inline]
    unsafe fn simd_load(ptr: *const Self) -> Self::Vector {
        core::ptr::read_unaligned(ptr as *const int64x2_t)
    }

    #[inline]
    unsafe fn simd_store(ptr: *mut Self, vec: Self::Vector) {
        core::ptr::write_unaligned(ptr as *mut int64x2_t, vec);
    }

    #[inline]
    unsafe fn simd_splat(val: Self) -> Self::Vector {
        vdupq_n_s64(val)
    }

    #[inline]
    unsafe fn simd_mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let a0 = vgetq_lane_s64(a, 0);
        let a1 = vgetq_lane_s64(a, 1);
        let b0 = vgetq_lane_s64(b, 0);
        let b1 = vgetq_lane_s64(b, 1);
        let mut result = vdupq_n_s64(0);
        result = vsetq_lane_s64(a0 * b0, result, 0);
        result = vsetq_lane_s64(a1 * b1, result, 1);
        result
    }

    #[inline]
    unsafe fn simd_add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        vaddq_s64(a, b)
    }

    #[inline]
    unsafe fn simd_fma(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        vaddq_s64(Self::simd_mul(a, b), c)
    }

    #[inline]
    unsafe fn simd_zero() -> Self::Vector {
        vdupq_n_s64(0)
    }

    #[inline]
    unsafe fn simd_horizontal_sum(vec: Self::Vector) -> Self {
        vaddvq_s64(vec)
    }
}

impl SimdType for u64 {
    type Vector = uint64x2_t;
    const LANES: usize = 2;

    #[inline]
    unsafe fn simd_load(ptr: *const Self) -> Self::Vector {
        core::ptr::read_unaligned(ptr as *const uint64x2_t)
    }

    #[inline]
    unsafe fn simd_store(ptr: *mut Self, vec: Self::Vector) {
        core::ptr::write_unaligned(ptr as *mut uint64x2_t, vec);
    }

    #[inline]
    unsafe fn simd_splat(val: Self) -> Self::Vector {
        vdupq_n_u64(val)
    }

    #[inline]
    unsafe fn simd_mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let a0 = vgetq_lane_u64(a, 0);
        let a1 = vgetq_lane_u64(a, 1);
        let b0 = vgetq_lane_u64(b, 0);
        let b1 = vgetq_lane_u64(b, 1);
        let mut result = vdupq_n_u64(0);
        result = vsetq_lane_u64(a0 * b0, result, 0);
        result = vsetq_lane_u64(a1 * b1, result, 1);
        result
    }

    #[inline]
    unsafe fn simd_add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        vaddq_u64(a, b)
    }

    #[inline]
    unsafe fn simd_fma(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        vaddq_u64(Self::simd_mul(a, b), c)
    }

    #[inline]
    unsafe fn simd_zero() -> Self::Vector {
        vdupq_n_u64(0)
    }

    #[inline]
    unsafe fn simd_horizontal_sum(vec: Self::Vector) -> Self {
        vaddvq_u64(vec)
    }
}
