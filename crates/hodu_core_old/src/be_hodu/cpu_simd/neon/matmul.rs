#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use core::arch::arm::*;

use crate::compat::*;

// Unified pack macro - handles both float and integer types
macro_rules! impl_pack {
    ($pack_a:ident, $pack_b:ident, $ty:ty, $mr:expr, $nr:expr, $zero:expr) => {
        unsafe fn $pack_a(
            src: &[$ty],
            dst: &mut [$ty],
            lda: usize,
            i_offset: usize,
            k_offset: usize,
            mc: usize,
            kc: usize,
        ) {
            const MR: usize = $mr;
            let mut dst_idx = 0;
            for i in (0..mc).step_by(MR) {
                let i_end = (i + MR).min(mc);
                for k in 0..kc {
                    for ii in i..i_end {
                        dst[dst_idx] = src[(i_offset + ii) * lda + k_offset + k];
                        dst_idx += 1;
                    }
                    dst[dst_idx..dst_idx + (i + MR - i_end)].fill($zero);
                    dst_idx += i + MR - i_end;
                }
            }
        }

        unsafe fn $pack_b(
            src: &[$ty],
            dst: &mut [$ty],
            ldb: usize,
            k_offset: usize,
            j_offset: usize,
            kc: usize,
            nc: usize,
        ) {
            const NR: usize = $nr;
            let mut dst_idx = 0;
            for j in (0..nc).step_by(NR) {
                let j_end = (j + NR).min(nc);
                for k in 0..kc {
                    for jj in j..j_end {
                        dst[dst_idx] = src[(k_offset + k) * ldb + j_offset + jj];
                        dst_idx += 1;
                    }
                    dst[dst_idx..dst_idx + (j + NR - j_end)].fill($zero);
                    dst_idx += j + NR - j_end;
                }
            }
        }
    };
}

// Generate pack functions (f32, f64, u8, u16, u32, i8, i16, i32)
impl_pack!(pack_a_f32, pack_b_f32, f32, 6, 4, 0.0f32);
impl_pack!(pack_a_f64, pack_b_f64, f64, 4, 2, 0.0f64);
impl_pack!(pack_a_u8, pack_b_u8, u8, 8, 16, 0u8);
impl_pack!(pack_a_u16, pack_b_u16, u16, 8, 8, 0u16);
impl_pack!(pack_a_u32, pack_b_u32, u32, 6, 4, 0u32);
impl_pack!(pack_a_i8, pack_b_i8, i8, 8, 16, 0i8);
impl_pack!(pack_a_i16, pack_b_i16, i16, 8, 8, 0i16);
impl_pack!(pack_a_i32, pack_b_i32, i32, 6, 4, 0i32);

// ============================================================================
// F32 Kernel (6x4) - NEON optimized
// ============================================================================
#[inline]
unsafe fn kernel_6x4_f32_packed(
    packed_a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    i_c: usize,
    j_c: usize,
    i_packed: usize,
    j_packed: usize,
    kc: usize,
    ldc: usize,
    m_actual: usize,
    n_actual: usize,
) {
    const MR: usize = 6;
    const NR: usize = 4;

    let m_edge = (i_c + MR).min(m_actual) - i_c;
    let n_edge = (j_c + NR).min(n_actual) - j_c;

    // Use SIMD only if we have full NR columns
    if n_edge == NR {
        let mut acc = [vdupq_n_f32(0.0); MR];
        for i in 0..m_edge {
            acc[i] = vld1q_f32(c.as_ptr().add((i_c + i) * ldc + j_c));
        }

        let a_panel = i_packed / MR;
        let b_panel = j_packed / NR;
        let mut a_ptr = a_panel * MR * kc;
        let mut b_ptr = b_panel * kc * NR;

        for _ in 0..kc {
            let b_vec = vld1q_f32(packed_b.as_ptr().add(b_ptr));
            for i in 0..m_edge {
                acc[i] = vfmaq_n_f32(acc[i], b_vec, *packed_a.get_unchecked(a_ptr + i));
            }
            a_ptr += MR;
            b_ptr += NR;
        }

        for i in 0..m_edge {
            vst1q_f32(c.as_mut_ptr().add((i_c + i) * ldc + j_c), acc[i]);
        }
    } else {
        // Scalar fallback for edge cases
        let a_panel = i_packed / MR;
        let b_panel = j_packed / NR;
        let a_base = a_panel * MR * kc;
        let b_base = b_panel * kc * NR;

        for i in 0..m_edge {
            for j in 0..n_edge {
                let mut sum = c[(i_c + i) * ldc + j_c + j];
                for k in 0..kc {
                    sum += packed_a[a_base + k * MR + i] * packed_b[b_base + k * NR + j];
                }
                c[(i_c + i) * ldc + j_c + j] = sum;
            }
        }
    }
}

// ============================================================================
// F64 Kernel (4x2) - NEON optimized
// ============================================================================
#[inline]
unsafe fn kernel_4x2_f64_packed(
    packed_a: &[f64],
    packed_b: &[f64],
    c: &mut [f64],
    i_c: usize,
    j_c: usize,
    i_packed: usize,
    j_packed: usize,
    kc: usize,
    ldc: usize,
    m_actual: usize,
    n_actual: usize,
) {
    const MR: usize = 4;
    const NR: usize = 2;

    let m_edge = (i_c + MR).min(m_actual) - i_c;
    let n_edge = (j_c + NR).min(n_actual) - j_c;

    if n_edge == NR {
        let mut acc = [vdupq_n_f64(0.0); MR];
        for i in 0..m_edge {
            acc[i] = vld1q_f64(c.as_ptr().add((i_c + i) * ldc + j_c));
        }

        let a_panel = i_packed / MR;
        let b_panel = j_packed / NR;
        let mut a_ptr = a_panel * MR * kc;
        let mut b_ptr = b_panel * kc * NR;

        for _ in 0..kc {
            let b_vec = vld1q_f64(packed_b.as_ptr().add(b_ptr));
            for i in 0..m_edge {
                acc[i] = vfmaq_n_f64(acc[i], b_vec, *packed_a.get_unchecked(a_ptr + i));
            }
            a_ptr += MR;
            b_ptr += NR;
        }

        for i in 0..m_edge {
            vst1q_f64(c.as_mut_ptr().add((i_c + i) * ldc + j_c), acc[i]);
        }
    } else {
        let a_panel = i_packed / MR;
        let b_panel = j_packed / NR;
        let a_base = a_panel * MR * kc;
        let b_base = b_panel * kc * NR;

        for i in 0..m_edge {
            for j in 0..n_edge {
                let mut sum = c[(i_c + i) * ldc + j_c + j];
                for k in 0..kc {
                    sum += packed_a[a_base + k * MR + i] * packed_b[b_base + k * NR + j];
                }
                c[(i_c + i) * ldc + j_c + j] = sum;
            }
        }
    }
}

// ============================================================================
// I8 Kernel (8x16) - NEON optimized
// ============================================================================
#[inline]
unsafe fn kernel_8x16_i8_packed(
    packed_a: &[i8],
    packed_b: &[i8],
    c: &mut [i8],
    i_c: usize,
    j_c: usize,
    i_packed: usize,
    j_packed: usize,
    kc: usize,
    ldc: usize,
    m_actual: usize,
    n_actual: usize,
) {
    const MR: usize = 8;
    const NR: usize = 16;

    let m_edge = (i_c + MR).min(m_actual) - i_c;
    let n_edge = (j_c + NR).min(n_actual) - j_c;

    let a_panel = i_packed / MR;
    let b_panel = j_packed / NR;
    let mut a_ptr = a_panel * MR * kc;
    let mut b_ptr = b_panel * kc * NR;

    let mut acc = [[0i16; NR]; MR];

    for _ in 0..kc {
        let mut b_vals = [0i8; NR];
        vst1q_s8(b_vals.as_mut_ptr(), vld1q_s8(packed_b.as_ptr().add(b_ptr)));

        for i in 0..m_edge {
            let a_val = *packed_a.get_unchecked(a_ptr + i) as i16;
            for j in 0..n_edge {
                acc[i][j] += a_val * b_vals[j] as i16;
            }
        }
        a_ptr += MR;
        b_ptr += NR;
    }

    for i in 0..m_edge {
        for j in 0..n_edge {
            c[(i_c + i) * ldc + j_c + j] = (c[(i_c + i) * ldc + j_c + j] as i16 + acc[i][j]).clamp(-128, 127) as i8;
        }
    }
}

// ============================================================================
// I16 Kernel (8x8) - NEON optimized
// ============================================================================
#[inline]
unsafe fn kernel_8x8_i16_packed(
    packed_a: &[i16],
    packed_b: &[i16],
    c: &mut [i16],
    i_c: usize,
    j_c: usize,
    i_packed: usize,
    j_packed: usize,
    kc: usize,
    ldc: usize,
    m_actual: usize,
    n_actual: usize,
) {
    const MR: usize = 8;
    const NR: usize = 8;

    let m_edge = (i_c + MR).min(m_actual) - i_c;
    let n_edge = (j_c + NR).min(n_actual) - j_c;

    let a_panel = i_packed / MR;
    let b_panel = j_packed / NR;
    let mut a_ptr = a_panel * MR * kc;
    let mut b_ptr = b_panel * kc * NR;

    let mut acc = [[0i32; NR]; MR];

    for _ in 0..kc {
        let mut b_vals = [0i16; NR];
        vst1q_s16(b_vals.as_mut_ptr(), vld1q_s16(packed_b.as_ptr().add(b_ptr)));

        for i in 0..m_edge {
            let a_val = *packed_a.get_unchecked(a_ptr + i) as i32;
            for j in 0..n_edge {
                acc[i][j] += a_val * b_vals[j] as i32;
            }
        }
        a_ptr += MR;
        b_ptr += NR;
    }

    for i in 0..m_edge {
        for j in 0..n_edge {
            c[(i_c + i) * ldc + j_c + j] =
                (c[(i_c + i) * ldc + j_c + j] as i32 + acc[i][j]).clamp(-32768, 32767) as i16;
        }
    }
}

// ============================================================================
// I32 Kernel (6x4) - NEON optimized
// ============================================================================
#[inline]
unsafe fn kernel_6x4_i32_packed(
    packed_a: &[i32],
    packed_b: &[i32],
    c: &mut [i32],
    i_c: usize,
    j_c: usize,
    i_packed: usize,
    j_packed: usize,
    kc: usize,
    ldc: usize,
    m_actual: usize,
    n_actual: usize,
) {
    const MR: usize = 6;
    const NR: usize = 4;

    let m_edge = (i_c + MR).min(m_actual) - i_c;
    let n_edge = (j_c + NR).min(n_actual) - j_c;

    if n_edge == NR {
        let mut acc = [vdupq_n_s32(0); MR];
        for i in 0..m_edge {
            acc[i] = vld1q_s32(c.as_ptr().add((i_c + i) * ldc + j_c));
        }

        let a_panel = i_packed / MR;
        let b_panel = j_packed / NR;
        let mut a_ptr = a_panel * MR * kc;
        let mut b_ptr = b_panel * kc * NR;

        for _ in 0..kc {
            let b_vec = vld1q_s32(packed_b.as_ptr().add(b_ptr));
            for i in 0..m_edge {
                let a_splat = vdupq_n_s32(*packed_a.get_unchecked(a_ptr + i));
                acc[i] = vmlaq_s32(acc[i], b_vec, a_splat);
            }
            a_ptr += MR;
            b_ptr += NR;
        }

        for i in 0..m_edge {
            vst1q_s32(c.as_mut_ptr().add((i_c + i) * ldc + j_c), acc[i]);
        }
    } else {
        let a_panel = i_packed / MR;
        let b_panel = j_packed / NR;
        let a_base = a_panel * MR * kc;
        let b_base = b_panel * kc * NR;

        for i in 0..m_edge {
            for j in 0..n_edge {
                let mut sum = c[(i_c + i) * ldc + j_c + j];
                for k in 0..kc {
                    sum += packed_a[a_base + k * MR + i] * packed_b[b_base + k * NR + j];
                }
                c[(i_c + i) * ldc + j_c + j] = sum;
            }
        }
    }
}

// ============================================================================
// U8 Kernel (8x16) - NEON optimized
// ============================================================================
#[inline]
unsafe fn kernel_8x16_u8_packed(
    packed_a: &[u8],
    packed_b: &[u8],
    c: &mut [u8],
    i_c: usize,
    j_c: usize,
    i_packed: usize,
    j_packed: usize,
    kc: usize,
    ldc: usize,
    m_actual: usize,
    n_actual: usize,
) {
    const MR: usize = 8;
    const NR: usize = 16;

    let m_edge = (i_c + MR).min(m_actual) - i_c;
    let n_edge = (j_c + NR).min(n_actual) - j_c;

    let a_panel = i_packed / MR;
    let b_panel = j_packed / NR;
    let mut a_ptr = a_panel * MR * kc;
    let mut b_ptr = b_panel * kc * NR;

    let mut acc = [[0u16; NR]; MR];

    for _ in 0..kc {
        let mut b_vals = [0u8; NR];
        vst1q_u8(b_vals.as_mut_ptr(), vld1q_u8(packed_b.as_ptr().add(b_ptr)));

        for i in 0..m_edge {
            let a_val = *packed_a.get_unchecked(a_ptr + i) as u16;
            for j in 0..n_edge {
                acc[i][j] += a_val * b_vals[j] as u16;
            }
        }
        a_ptr += MR;
        b_ptr += NR;
    }

    for i in 0..m_edge {
        for j in 0..n_edge {
            c[(i_c + i) * ldc + j_c + j] = (c[(i_c + i) * ldc + j_c + j] as u16 + acc[i][j]).min(255) as u8;
        }
    }
}

// ============================================================================
// U16 Kernel (8x8) - NEON optimized
// ============================================================================
#[inline]
unsafe fn kernel_8x8_u16_packed(
    packed_a: &[u16],
    packed_b: &[u16],
    c: &mut [u16],
    i_c: usize,
    j_c: usize,
    i_packed: usize,
    j_packed: usize,
    kc: usize,
    ldc: usize,
    m_actual: usize,
    n_actual: usize,
) {
    const MR: usize = 8;
    const NR: usize = 8;

    let m_edge = (i_c + MR).min(m_actual) - i_c;
    let n_edge = (j_c + NR).min(n_actual) - j_c;

    let a_panel = i_packed / MR;
    let b_panel = j_packed / NR;
    let mut a_ptr = a_panel * MR * kc;
    let mut b_ptr = b_panel * kc * NR;

    let mut acc = [[0u32; NR]; MR];

    for _ in 0..kc {
        let mut b_vals = [0u16; NR];
        vst1q_u16(b_vals.as_mut_ptr(), vld1q_u16(packed_b.as_ptr().add(b_ptr)));

        for i in 0..m_edge {
            let a_val = *packed_a.get_unchecked(a_ptr + i) as u32;
            for j in 0..n_edge {
                acc[i][j] += a_val * b_vals[j] as u32;
            }
        }
        a_ptr += MR;
        b_ptr += NR;
    }

    for i in 0..m_edge {
        for j in 0..n_edge {
            c[(i_c + i) * ldc + j_c + j] = (c[(i_c + i) * ldc + j_c + j] as u32 + acc[i][j]).min(65535) as u16;
        }
    }
}

// ============================================================================
// U32 Kernel (6x4) - NEON optimized
// ============================================================================
#[inline]
unsafe fn kernel_6x4_u32_packed(
    packed_a: &[u32],
    packed_b: &[u32],
    c: &mut [u32],
    i_c: usize,
    j_c: usize,
    i_packed: usize,
    j_packed: usize,
    kc: usize,
    ldc: usize,
    m_actual: usize,
    n_actual: usize,
) {
    const MR: usize = 6;
    const NR: usize = 4;

    let m_edge = (i_c + MR).min(m_actual) - i_c;
    let n_edge = (j_c + NR).min(n_actual) - j_c;

    if n_edge == NR {
        let mut acc = [vdupq_n_u32(0); MR];
        for i in 0..m_edge {
            acc[i] = vld1q_u32(c.as_ptr().add((i_c + i) * ldc + j_c));
        }

        let a_panel = i_packed / MR;
        let b_panel = j_packed / NR;
        let mut a_ptr = a_panel * MR * kc;
        let mut b_ptr = b_panel * kc * NR;

        for _ in 0..kc {
            let b_vec = vld1q_u32(packed_b.as_ptr().add(b_ptr));
            for i in 0..m_edge {
                let a_splat = vdupq_n_u32(*packed_a.get_unchecked(a_ptr + i));
                acc[i] = vmlaq_u32(acc[i], b_vec, a_splat);
            }
            a_ptr += MR;
            b_ptr += NR;
        }

        for i in 0..m_edge {
            vst1q_u32(c.as_mut_ptr().add((i_c + i) * ldc + j_c), acc[i]);
        }
    } else {
        let a_panel = i_packed / MR;
        let b_panel = j_packed / NR;
        let a_base = a_panel * MR * kc;
        let b_base = b_panel * kc * NR;

        for i in 0..m_edge {
            for j in 0..n_edge {
                let mut sum = c[(i_c + i) * ldc + j_c + j];
                for k in 0..kc {
                    sum += packed_a[a_base + k * MR + i] * packed_b[b_base + k * NR + j];
                }
                c[(i_c + i) * ldc + j_c + j] = sum;
            }
        }
    }
}

// ============================================================================
// GEMM implementation macros - split into rayon and sequential versions
// ============================================================================

#[cfg(feature = "rayon")]
macro_rules! impl_gemm_rayon {
    ($fn_name:ident, $ty:ty, $pack_a:ident, $pack_b:ident, $kernel:ident, $mr:expr, $nr:expr, $zero:expr) => {
        pub unsafe fn $fn_name(a: &[$ty], b: &[$ty], c: &mut [$ty], m: usize, k: usize, n: usize) {
            use rayon::prelude::*;

            const MC: usize = 256;
            const KC: usize = 256;
            const NC: usize = 2048;
            const MR: usize = $mr;
            const NR: usize = $nr;

            for jc in (0..n).step_by(NC) {
                let nc = (jc + NC).min(n) - jc;
                for pc in (0..k).step_by(KC) {
                    let kc = (pc + KC).min(k) - pc;
                    let packed_b_size = kc * nc.div_ceil(NR) * NR;
                    let mut packed_b = vec![$zero; packed_b_size];
                    $pack_b(b, &mut packed_b, n, pc, jc, kc, nc);

                    let c_addr = c.as_mut_ptr() as usize;
                    let c_len = c.len();
                    let ic_indices: Vec<usize> = (0..m).step_by(MC).collect();
                    ic_indices.into_par_iter().for_each(|ic| {
                        let c_ptr = c_addr as *mut $ty;
                        let mc = (ic + MC).min(m) - ic;
                        let packed_a_size = mc.div_ceil(MR) * MR * kc;
                        let mut packed_a = vec![$zero; packed_a_size];
                        $pack_a(a, &mut packed_a, k, ic, pc, mc, kc);
                        for ir in (0..mc).step_by(MR) {
                            for jr in (0..nc).step_by(NR) {
                                $kernel(
                                    &packed_a,
                                    &packed_b,
                                    core::slice::from_raw_parts_mut(c_ptr, c_len),
                                    ic + ir,
                                    jc + jr,
                                    ir,
                                    jr,
                                    kc,
                                    n,
                                    m,
                                    n,
                                );
                            }
                        }
                    });
                }
            }
        }
    };
}

#[cfg(not(feature = "rayon"))]
macro_rules! impl_gemm_seq {
    ($fn_name:ident, $ty:ty, $pack_a:ident, $pack_b:ident, $kernel:ident, $mr:expr, $nr:expr, $zero:expr) => {
        pub unsafe fn $fn_name(a: &[$ty], b: &[$ty], c: &mut [$ty], m: usize, k: usize, n: usize) {
            const MC: usize = 256;
            const KC: usize = 256;
            const NC: usize = 2048;
            const MR: usize = $mr;
            const NR: usize = $nr;

            for jc in (0..n).step_by(NC) {
                let nc = (jc + NC).min(n) - jc;
                for pc in (0..k).step_by(KC) {
                    let kc = (pc + KC).min(k) - pc;
                    let packed_b_size = kc * nc.div_ceil(NR) * NR;
                    let mut packed_b = vec![$zero; packed_b_size];
                    $pack_b(b, &mut packed_b, n, pc, jc, kc, nc);

                    for ic in (0..m).step_by(MC) {
                        let mc = (ic + MC).min(m) - ic;
                        let packed_a_size = mc.div_ceil(MR) * MR * kc;
                        let mut packed_a = vec![$zero; packed_a_size];
                        $pack_a(a, &mut packed_a, k, ic, pc, mc, kc);
                        for ir in (0..mc).step_by(MR) {
                            for jr in (0..nc).step_by(NR) {
                                $kernel(&packed_a, &packed_b, c, ic + ir, jc + jr, ir, jr, kc, n, m, n);
                            }
                        }
                    }
                }
            }
        }
    };
}

#[cfg(feature = "rayon")]
impl_gemm_rayon!(f32, f32, pack_a_f32, pack_b_f32, kernel_6x4_f32_packed, 6, 4, 0.0f32);
#[cfg(not(feature = "rayon"))]
impl_gemm_seq!(f32, f32, pack_a_f32, pack_b_f32, kernel_6x4_f32_packed, 6, 4, 0.0f32);

#[cfg(feature = "rayon")]
impl_gemm_rayon!(f64, f64, pack_a_f64, pack_b_f64, kernel_4x2_f64_packed, 4, 2, 0.0f64);
#[cfg(not(feature = "rayon"))]
impl_gemm_seq!(f64, f64, pack_a_f64, pack_b_f64, kernel_4x2_f64_packed, 4, 2, 0.0f64);

#[cfg(feature = "rayon")]
impl_gemm_rayon!(u8, u8, pack_a_u8, pack_b_u8, kernel_8x16_u8_packed, 8, 16, 0u8);
#[cfg(not(feature = "rayon"))]
impl_gemm_seq!(u8, u8, pack_a_u8, pack_b_u8, kernel_8x16_u8_packed, 8, 16, 0u8);

#[cfg(feature = "rayon")]
impl_gemm_rayon!(u16, u16, pack_a_u16, pack_b_u16, kernel_8x8_u16_packed, 8, 8, 0u16);
#[cfg(not(feature = "rayon"))]
impl_gemm_seq!(u16, u16, pack_a_u16, pack_b_u16, kernel_8x8_u16_packed, 8, 8, 0u16);

#[cfg(feature = "rayon")]
impl_gemm_rayon!(u32, u32, pack_a_u32, pack_b_u32, kernel_6x4_u32_packed, 6, 4, 0u32);
#[cfg(not(feature = "rayon"))]
impl_gemm_seq!(u32, u32, pack_a_u32, pack_b_u32, kernel_6x4_u32_packed, 6, 4, 0u32);

#[cfg(feature = "rayon")]
impl_gemm_rayon!(i8, i8, pack_a_i8, pack_b_i8, kernel_8x16_i8_packed, 8, 16, 0i8);
#[cfg(not(feature = "rayon"))]
impl_gemm_seq!(i8, i8, pack_a_i8, pack_b_i8, kernel_8x16_i8_packed, 8, 16, 0i8);

#[cfg(feature = "rayon")]
impl_gemm_rayon!(i16, i16, pack_a_i16, pack_b_i16, kernel_8x8_i16_packed, 8, 8, 0i16);
#[cfg(not(feature = "rayon"))]
impl_gemm_seq!(i16, i16, pack_a_i16, pack_b_i16, kernel_8x8_i16_packed, 8, 8, 0i16);

#[cfg(feature = "rayon")]
impl_gemm_rayon!(i32, i32, pack_a_i32, pack_b_i32, kernel_6x4_i32_packed, 6, 4, 0i32);
#[cfg(not(feature = "rayon"))]
impl_gemm_seq!(i32, i32, pack_a_i32, pack_b_i32, kernel_6x4_i32_packed, 6, 4, 0i32);
