#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use core::arch::arm::*;

use crate::compat::*;

unsafe fn pack_a_f32(src: &[f32], dst: &mut [f32], lda: usize, i_offset: usize, k_offset: usize, mc: usize, kc: usize) {
    const MR: usize = 6;
    let mut dst_idx = 0;

    for i in (0..mc).step_by(MR) {
        let i_end = (i + MR).min(mc);
        for k in 0..kc {
            for ii in i..i_end {
                dst[dst_idx] = src[(i_offset + ii) * lda + k_offset + k];
                dst_idx += 1;
            }
            for _ in i_end..(i + MR) {
                dst[dst_idx] = 0.0;
                dst_idx += 1;
            }
        }
    }
}

unsafe fn pack_b_f32(src: &[f32], dst: &mut [f32], ldb: usize, k_offset: usize, j_offset: usize, kc: usize, nc: usize) {
    const NR: usize = 4;
    let mut dst_idx = 0;

    for j in (0..nc).step_by(NR) {
        let j_end = (j + NR).min(nc);
        for k in 0..kc {
            for jj in j..j_end {
                dst[dst_idx] = src[(k_offset + k) * ldb + j_offset + jj];
                dst_idx += 1;
            }
            for _ in j_end..(j + NR) {
                dst[dst_idx] = 0.0;
                dst_idx += 1;
            }
        }
    }
}

unsafe fn pack_a_f64(src: &[f64], dst: &mut [f64], lda: usize, i_offset: usize, k_offset: usize, mc: usize, kc: usize) {
    const MR: usize = 4;
    let mut dst_idx = 0;

    for i in (0..mc).step_by(MR) {
        let i_end = (i + MR).min(mc);
        for k in 0..kc {
            for ii in i..i_end {
                dst[dst_idx] = src[(i_offset + ii) * lda + k_offset + k];
                dst_idx += 1;
            }
            for _ in i_end..(i + MR) {
                dst[dst_idx] = 0.0;
                dst_idx += 1;
            }
        }
    }
}

unsafe fn pack_b_f64(src: &[f64], dst: &mut [f64], ldb: usize, k_offset: usize, j_offset: usize, kc: usize, nc: usize) {
    const NR: usize = 2;
    let mut dst_idx = 0;

    for j in (0..nc).step_by(NR) {
        let j_end = (j + NR).min(nc);
        for k in 0..kc {
            for jj in j..j_end {
                dst[dst_idx] = src[(k_offset + k) * ldb + j_offset + jj];
                dst_idx += 1;
            }
            for _ in j_end..(j + NR) {
                dst[dst_idx] = 0.0;
                dst_idx += 1;
            }
        }
    }
}

#[inline(always)]
unsafe fn kernel_6x4_f32_blocked(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    i: usize,
    j: usize,
    k_start: usize,
    k_len: usize,
    k: usize,
    n: usize,
) {
    // const MR: usize = 6;
    // const NR: usize = 4;

    let mut c0 = vld1q_f32(c.as_ptr().add(i * n + j));
    let mut c1 = vld1q_f32(c.as_ptr().add((i + 1) * n + j));
    let mut c2 = vld1q_f32(c.as_ptr().add((i + 2) * n + j));
    let mut c3 = vld1q_f32(c.as_ptr().add((i + 3) * n + j));
    let mut c4 = vld1q_f32(c.as_ptr().add((i + 4) * n + j));
    let mut c5 = vld1q_f32(c.as_ptr().add((i + 5) * n + j));

    for kk in 0..k_len {
        let k_idx = k_start + kk;
        let a0 = *a.get_unchecked(i * k + k_idx);
        let a1 = *a.get_unchecked((i + 1) * k + k_idx);
        let a2 = *a.get_unchecked((i + 2) * k + k_idx);
        let a3 = *a.get_unchecked((i + 3) * k + k_idx);
        let a4 = *a.get_unchecked((i + 4) * k + k_idx);
        let a5 = *a.get_unchecked((i + 5) * k + k_idx);
        let b_vec = vld1q_f32(b.as_ptr().add(k_idx * n + j));

        c0 = vfmaq_n_f32(c0, b_vec, a0);
        c1 = vfmaq_n_f32(c1, b_vec, a1);
        c2 = vfmaq_n_f32(c2, b_vec, a2);
        c3 = vfmaq_n_f32(c3, b_vec, a3);
        c4 = vfmaq_n_f32(c4, b_vec, a4);
        c5 = vfmaq_n_f32(c5, b_vec, a5);
    }

    vst1q_f32(c.as_mut_ptr().add(i * n + j), c0);
    vst1q_f32(c.as_mut_ptr().add((i + 1) * n + j), c1);
    vst1q_f32(c.as_mut_ptr().add((i + 2) * n + j), c2);
    vst1q_f32(c.as_mut_ptr().add((i + 3) * n + j), c3);
    vst1q_f32(c.as_mut_ptr().add((i + 4) * n + j), c4);
    vst1q_f32(c.as_mut_ptr().add((i + 5) * n + j), c5);
}

#[inline(always)]
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
) {
    const MR: usize = 6;
    const NR: usize = 4;

    let mut c0 = vld1q_f32(c.as_ptr().add(i_c * ldc + j_c));
    let mut c1 = vld1q_f32(c.as_ptr().add((i_c + 1) * ldc + j_c));
    let mut c2 = vld1q_f32(c.as_ptr().add((i_c + 2) * ldc + j_c));
    let mut c3 = vld1q_f32(c.as_ptr().add((i_c + 3) * ldc + j_c));
    let mut c4 = vld1q_f32(c.as_ptr().add((i_c + 4) * ldc + j_c));
    let mut c5 = vld1q_f32(c.as_ptr().add((i_c + 5) * ldc + j_c));

    let a_panel = i_packed / MR;
    let b_panel = j_packed / NR;
    let mut a_ptr = a_panel * MR * kc;
    let mut b_ptr = b_panel * kc * NR;

    let k_unroll = kc / 4 * 4;

    for _ in (0..k_unroll).step_by(4) {
        let a0_0 = *packed_a.get_unchecked(a_ptr);
        let a1_0 = *packed_a.get_unchecked(a_ptr + 1);
        let a2_0 = *packed_a.get_unchecked(a_ptr + 2);
        let a3_0 = *packed_a.get_unchecked(a_ptr + 3);
        let a4_0 = *packed_a.get_unchecked(a_ptr + 4);
        let a5_0 = *packed_a.get_unchecked(a_ptr + 5);
        let b_vec0 = vld1q_f32(packed_b.as_ptr().add(b_ptr));
        a_ptr += MR;
        b_ptr += NR;

        let a0_1 = *packed_a.get_unchecked(a_ptr);
        let a1_1 = *packed_a.get_unchecked(a_ptr + 1);
        let a2_1 = *packed_a.get_unchecked(a_ptr + 2);
        let a3_1 = *packed_a.get_unchecked(a_ptr + 3);
        let a4_1 = *packed_a.get_unchecked(a_ptr + 4);
        let a5_1 = *packed_a.get_unchecked(a_ptr + 5);
        let b_vec1 = vld1q_f32(packed_b.as_ptr().add(b_ptr));
        a_ptr += MR;
        b_ptr += NR;

        let a0_2 = *packed_a.get_unchecked(a_ptr);
        let a1_2 = *packed_a.get_unchecked(a_ptr + 1);
        let a2_2 = *packed_a.get_unchecked(a_ptr + 2);
        let a3_2 = *packed_a.get_unchecked(a_ptr + 3);
        let a4_2 = *packed_a.get_unchecked(a_ptr + 4);
        let a5_2 = *packed_a.get_unchecked(a_ptr + 5);
        let b_vec2 = vld1q_f32(packed_b.as_ptr().add(b_ptr));
        a_ptr += MR;
        b_ptr += NR;

        let a0_3 = *packed_a.get_unchecked(a_ptr);
        let a1_3 = *packed_a.get_unchecked(a_ptr + 1);
        let a2_3 = *packed_a.get_unchecked(a_ptr + 2);
        let a3_3 = *packed_a.get_unchecked(a_ptr + 3);
        let a4_3 = *packed_a.get_unchecked(a_ptr + 4);
        let a5_3 = *packed_a.get_unchecked(a_ptr + 5);
        let b_vec3 = vld1q_f32(packed_b.as_ptr().add(b_ptr));
        a_ptr += MR;
        b_ptr += NR;

        c0 = vfmaq_n_f32(c0, b_vec0, a0_0);
        c1 = vfmaq_n_f32(c1, b_vec0, a1_0);
        c2 = vfmaq_n_f32(c2, b_vec0, a2_0);
        c3 = vfmaq_n_f32(c3, b_vec0, a3_0);
        c4 = vfmaq_n_f32(c4, b_vec0, a4_0);
        c5 = vfmaq_n_f32(c5, b_vec0, a5_0);

        c0 = vfmaq_n_f32(c0, b_vec1, a0_1);
        c1 = vfmaq_n_f32(c1, b_vec1, a1_1);
        c2 = vfmaq_n_f32(c2, b_vec1, a2_1);
        c3 = vfmaq_n_f32(c3, b_vec1, a3_1);
        c4 = vfmaq_n_f32(c4, b_vec1, a4_1);
        c5 = vfmaq_n_f32(c5, b_vec1, a5_1);

        c0 = vfmaq_n_f32(c0, b_vec2, a0_2);
        c1 = vfmaq_n_f32(c1, b_vec2, a1_2);
        c2 = vfmaq_n_f32(c2, b_vec2, a2_2);
        c3 = vfmaq_n_f32(c3, b_vec2, a3_2);
        c4 = vfmaq_n_f32(c4, b_vec2, a4_2);
        c5 = vfmaq_n_f32(c5, b_vec2, a5_2);

        c0 = vfmaq_n_f32(c0, b_vec3, a0_3);
        c1 = vfmaq_n_f32(c1, b_vec3, a1_3);
        c2 = vfmaq_n_f32(c2, b_vec3, a2_3);
        c3 = vfmaq_n_f32(c3, b_vec3, a3_3);
        c4 = vfmaq_n_f32(c4, b_vec3, a4_3);
        c5 = vfmaq_n_f32(c5, b_vec3, a5_3);
    }

    for _ in k_unroll..kc {
        let a0 = *packed_a.get_unchecked(a_ptr);
        let a1 = *packed_a.get_unchecked(a_ptr + 1);
        let a2 = *packed_a.get_unchecked(a_ptr + 2);
        let a3 = *packed_a.get_unchecked(a_ptr + 3);
        let a4 = *packed_a.get_unchecked(a_ptr + 4);
        let a5 = *packed_a.get_unchecked(a_ptr + 5);
        let b_vec = vld1q_f32(packed_b.as_ptr().add(b_ptr));
        a_ptr += MR;
        b_ptr += NR;

        c0 = vfmaq_n_f32(c0, b_vec, a0);
        c1 = vfmaq_n_f32(c1, b_vec, a1);
        c2 = vfmaq_n_f32(c2, b_vec, a2);
        c3 = vfmaq_n_f32(c3, b_vec, a3);
        c4 = vfmaq_n_f32(c4, b_vec, a4);
        c5 = vfmaq_n_f32(c5, b_vec, a5);
    }

    vst1q_f32(c.as_mut_ptr().add(i_c * ldc + j_c), c0);
    vst1q_f32(c.as_mut_ptr().add((i_c + 1) * ldc + j_c), c1);
    vst1q_f32(c.as_mut_ptr().add((i_c + 2) * ldc + j_c), c2);
    vst1q_f32(c.as_mut_ptr().add((i_c + 3) * ldc + j_c), c3);
    vst1q_f32(c.as_mut_ptr().add((i_c + 4) * ldc + j_c), c4);
    vst1q_f32(c.as_mut_ptr().add((i_c + 5) * ldc + j_c), c5);
}

#[inline(always)]
unsafe fn kernel_4x2_f64_blocked(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    i: usize,
    j: usize,
    k_start: usize,
    k_len: usize,
    k: usize,
    n: usize,
) {
    // const MR: usize = 4;
    // const NR: usize = 2;

    let mut c0 = vld1q_f64(c.as_ptr().add(i * n + j));
    let mut c1 = vld1q_f64(c.as_ptr().add((i + 1) * n + j));
    let mut c2 = vld1q_f64(c.as_ptr().add((i + 2) * n + j));
    let mut c3 = vld1q_f64(c.as_ptr().add((i + 3) * n + j));

    for kk in 0..k_len {
        let k_idx = k_start + kk;
        let a0 = *a.get_unchecked(i * k + k_idx);
        let a1 = *a.get_unchecked((i + 1) * k + k_idx);
        let a2 = *a.get_unchecked((i + 2) * k + k_idx);
        let a3 = *a.get_unchecked((i + 3) * k + k_idx);
        let b_vec = vld1q_f64(b.as_ptr().add(k_idx * n + j));

        c0 = vfmaq_n_f64(c0, b_vec, a0);
        c1 = vfmaq_n_f64(c1, b_vec, a1);
        c2 = vfmaq_n_f64(c2, b_vec, a2);
        c3 = vfmaq_n_f64(c3, b_vec, a3);
    }

    vst1q_f64(c.as_mut_ptr().add(i * n + j), c0);
    vst1q_f64(c.as_mut_ptr().add((i + 1) * n + j), c1);
    vst1q_f64(c.as_mut_ptr().add((i + 2) * n + j), c2);
    vst1q_f64(c.as_mut_ptr().add((i + 3) * n + j), c3);
}

#[inline(always)]
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
) {
    const MR: usize = 4;
    const NR: usize = 2;

    let mut c0 = vld1q_f64(c.as_ptr().add(i_c * ldc + j_c));
    let mut c1 = vld1q_f64(c.as_ptr().add((i_c + 1) * ldc + j_c));
    let mut c2 = vld1q_f64(c.as_ptr().add((i_c + 2) * ldc + j_c));
    let mut c3 = vld1q_f64(c.as_ptr().add((i_c + 3) * ldc + j_c));

    let a_panel = i_packed / MR;
    let b_panel = j_packed / NR;
    let mut a_ptr = a_panel * MR * kc;
    let mut b_ptr = b_panel * kc * NR;

    let k_unroll = kc / 4 * 4;

    for _ in (0..k_unroll).step_by(4) {
        let a0_0 = *packed_a.get_unchecked(a_ptr);
        let a1_0 = *packed_a.get_unchecked(a_ptr + 1);
        let a2_0 = *packed_a.get_unchecked(a_ptr + 2);
        let a3_0 = *packed_a.get_unchecked(a_ptr + 3);
        let b_vec0 = vld1q_f64(packed_b.as_ptr().add(b_ptr));
        a_ptr += MR;
        b_ptr += NR;

        let a0_1 = *packed_a.get_unchecked(a_ptr);
        let a1_1 = *packed_a.get_unchecked(a_ptr + 1);
        let a2_1 = *packed_a.get_unchecked(a_ptr + 2);
        let a3_1 = *packed_a.get_unchecked(a_ptr + 3);
        let b_vec1 = vld1q_f64(packed_b.as_ptr().add(b_ptr));
        a_ptr += MR;
        b_ptr += NR;

        let a0_2 = *packed_a.get_unchecked(a_ptr);
        let a1_2 = *packed_a.get_unchecked(a_ptr + 1);
        let a2_2 = *packed_a.get_unchecked(a_ptr + 2);
        let a3_2 = *packed_a.get_unchecked(a_ptr + 3);
        let b_vec2 = vld1q_f64(packed_b.as_ptr().add(b_ptr));
        a_ptr += MR;
        b_ptr += NR;

        let a0_3 = *packed_a.get_unchecked(a_ptr);
        let a1_3 = *packed_a.get_unchecked(a_ptr + 1);
        let a2_3 = *packed_a.get_unchecked(a_ptr + 2);
        let a3_3 = *packed_a.get_unchecked(a_ptr + 3);
        let b_vec3 = vld1q_f64(packed_b.as_ptr().add(b_ptr));
        a_ptr += MR;
        b_ptr += NR;

        c0 = vfmaq_n_f64(c0, b_vec0, a0_0);
        c1 = vfmaq_n_f64(c1, b_vec0, a1_0);
        c2 = vfmaq_n_f64(c2, b_vec0, a2_0);
        c3 = vfmaq_n_f64(c3, b_vec0, a3_0);

        c0 = vfmaq_n_f64(c0, b_vec1, a0_1);
        c1 = vfmaq_n_f64(c1, b_vec1, a1_1);
        c2 = vfmaq_n_f64(c2, b_vec1, a2_1);
        c3 = vfmaq_n_f64(c3, b_vec1, a3_1);

        c0 = vfmaq_n_f64(c0, b_vec2, a0_2);
        c1 = vfmaq_n_f64(c1, b_vec2, a1_2);
        c2 = vfmaq_n_f64(c2, b_vec2, a2_2);
        c3 = vfmaq_n_f64(c3, b_vec2, a3_2);

        c0 = vfmaq_n_f64(c0, b_vec3, a0_3);
        c1 = vfmaq_n_f64(c1, b_vec3, a1_3);
        c2 = vfmaq_n_f64(c2, b_vec3, a2_3);
        c3 = vfmaq_n_f64(c3, b_vec3, a3_3);
    }

    for _ in k_unroll..kc {
        let a0 = *packed_a.get_unchecked(a_ptr);
        let a1 = *packed_a.get_unchecked(a_ptr + 1);
        let a2 = *packed_a.get_unchecked(a_ptr + 2);
        let a3 = *packed_a.get_unchecked(a_ptr + 3);
        let b_vec = vld1q_f64(packed_b.as_ptr().add(b_ptr));
        a_ptr += MR;
        b_ptr += NR;

        c0 = vfmaq_n_f64(c0, b_vec, a0);
        c1 = vfmaq_n_f64(c1, b_vec, a1);
        c2 = vfmaq_n_f64(c2, b_vec, a2);
        c3 = vfmaq_n_f64(c3, b_vec, a3);
    }

    vst1q_f64(c.as_mut_ptr().add(i_c * ldc + j_c), c0);
    vst1q_f64(c.as_mut_ptr().add((i_c + 1) * ldc + j_c), c1);
    vst1q_f64(c.as_mut_ptr().add((i_c + 2) * ldc + j_c), c2);
    vst1q_f64(c.as_mut_ptr().add((i_c + 3) * ldc + j_c), c3);
}

unsafe fn f32_blocked_no_pack(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    const MC: usize = 256;
    const KC: usize = 256;
    const NC: usize = 2048;
    const MR: usize = 6;
    const NR: usize = 4;

    for jc in (0..n).step_by(NC) {
        let nc = (jc + NC).min(n);

        for pc in (0..k).step_by(KC) {
            let kc = (pc + KC).min(k);
            let k_len = kc - pc;

            #[cfg(feature = "rayon")]
            {
                let c_addr = c.as_mut_ptr() as usize;
                let ic_indices: Vec<usize> = (0..m).step_by(MC).collect();

                ic_indices.into_par_iter().for_each(|ic| {
                    let c_ptr = c_addr as *mut f32;
                    let mc = (ic + MC).min(m);

                    for i in (ic..mc).step_by(MR) {
                        for j in (jc..nc).step_by(NR) {
                            let i_end = (i + MR).min(mc);
                            let j_end = (j + NR).min(nc);

                            if i_end == i + MR && j_end == j + NR {
                                if pc == 0 {
                                    for ii in i..i_end {
                                        for jj in j..j_end {
                                            *c_ptr.add(ii * n + jj) = 0.0;
                                        }
                                    }
                                }
                                kernel_6x4_f32_blocked(
                                    a,
                                    b,
                                    core::slice::from_raw_parts_mut(c_ptr, c.len()),
                                    i,
                                    j,
                                    pc,
                                    k_len,
                                    k,
                                    n,
                                );
                            } else {
                                for ii in i..i_end {
                                    for jj in j..j_end {
                                        if pc == 0 {
                                            *c_ptr.add(ii * n + jj) = 0.0;
                                        }
                                        for kk in pc..kc {
                                            *c_ptr.add(ii * n + jj) += a[ii * k + kk] * b[kk * n + jj];
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }

            #[cfg(not(feature = "rayon"))]
            {
                for ic in (0..m).step_by(MC) {
                    let mc = (ic + MC).min(m);

                    for i in (ic..mc).step_by(MR) {
                        for j in (jc..nc).step_by(NR) {
                            let i_end = (i + MR).min(mc);
                            let j_end = (j + NR).min(nc);

                            if i_end == i + MR && j_end == j + NR {
                                if pc == 0 {
                                    for ii in i..i_end {
                                        for jj in j..j_end {
                                            c[ii * n + jj] = 0.0;
                                        }
                                    }
                                }
                                kernel_6x4_f32_blocked(a, b, c, i, j, pc, k_len, k, n);
                            } else {
                                for ii in i..i_end {
                                    for jj in j..j_end {
                                        if pc == 0 {
                                            c[ii * n + jj] = 0.0;
                                        }
                                        for kk in pc..kc {
                                            c[ii * n + jj] += a[ii * k + kk] * b[kk * n + jj];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub unsafe fn f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    const MC: usize = 256;
    const KC: usize = 256;
    const NC: usize = 2048;
    const MR: usize = 6;
    const NR: usize = 4;

    let use_packing = m >= 512 && n >= 512 && k >= 512;

    if !use_packing {
        return f32_blocked_no_pack(a, b, c, m, k, n);
    }

    for jc in (0..n).step_by(NC) {
        let nc = (jc + NC).min(n) - jc;

        for pc in (0..k).step_by(KC) {
            let kc_val = (pc + KC).min(k) - pc;

            let packed_b_size = kc_val * ((nc + NR - 1) / NR) * NR;
            let mut packed_b = vec![0.0f32; packed_b_size];

            pack_b_f32(b, &mut packed_b, n, pc, jc, kc_val, nc);

            #[cfg(feature = "rayon")]
            {
                let c_addr = c.as_mut_ptr() as usize;
                let c_len = c.len();
                let ic_indices: Vec<usize> = (0..m).step_by(MC).collect();

                ic_indices.into_par_iter().for_each(|ic| {
                    let c_ptr = c_addr as *mut f32;
                    let mc = (ic + MC).min(m) - ic;
                    let packed_a_size = ((mc + MR - 1) / MR) * MR * kc_val;
                    let mut packed_a = vec![0.0f32; packed_a_size];

                    pack_a_f32(a, &mut packed_a, k, ic, pc, mc, kc_val);

                    for i in (0..mc).step_by(MR) {
                        for j in (0..nc).step_by(NR) {
                            let i_end = (i + MR).min(mc);
                            let j_end = (j + NR).min(nc);

                            if i_end == i + MR && j_end == j + NR {
                                if pc == 0 {
                                    for ii in 0..MR {
                                        for jj in 0..NR {
                                            *c_ptr.add((ic + i + ii) * n + (jc + j + jj)) = 0.0;
                                        }
                                    }
                                }
                                kernel_6x4_f32_packed(
                                    &packed_a,
                                    &packed_b,
                                    core::slice::from_raw_parts_mut(c_ptr, c_len),
                                    ic + i,
                                    jc + j,
                                    i,
                                    j,
                                    kc_val,
                                    n,
                                );
                            } else {
                                for ii in i..i_end {
                                    for jj in j..j_end {
                                        if pc == 0 {
                                            *c_ptr.add((ic + ii) * n + (jc + jj)) = 0.0;
                                        }
                                        for kk in 0..kc_val {
                                            *c_ptr.add((ic + ii) * n + (jc + jj)) +=
                                                a[(ic + ii) * k + pc + kk] * b[(pc + kk) * n + jc + jj];
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }

            #[cfg(not(feature = "rayon"))]
            {
                for ic in (0..m).step_by(MC) {
                    let mc = (ic + MC).min(m) - ic;
                    let packed_a_size = ((mc + MR - 1) / MR) * MR * kc_val;
                    let mut packed_a = vec![0.0f32; packed_a_size];

                    pack_a_f32(a, &mut packed_a, k, ic, pc, mc, kc_val);

                    for i in (0..mc).step_by(MR) {
                        for j in (0..nc).step_by(NR) {
                            let i_end = (i + MR).min(mc);
                            let j_end = (j + NR).min(nc);

                            if i_end == i + MR && j_end == j + NR {
                                if pc == 0 {
                                    for ii in 0..MR {
                                        for jj in 0..NR {
                                            c[(ic + i + ii) * n + (jc + j + jj)] = 0.0;
                                        }
                                    }
                                }
                                kernel_6x4_f32_packed(&packed_a, &packed_b, c, ic + i, jc + j, i, j, kc_val, n);
                            } else {
                                for ii in i..i_end {
                                    for jj in j..j_end {
                                        if pc == 0 {
                                            c[(ic + ii) * n + (jc + jj)] = 0.0;
                                        }
                                        for kk in 0..kc_val {
                                            c[(ic + ii) * n + (jc + jj)] +=
                                                a[(ic + ii) * k + pc + kk] * b[(pc + kk) * n + jc + jj];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

unsafe fn f64_blocked_no_pack(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    const MC: usize = 256;
    const KC: usize = 256;
    const NC: usize = 2048;
    const MR: usize = 4;
    const NR: usize = 2;

    for jc in (0..n).step_by(NC) {
        let nc = (jc + NC).min(n);

        for pc in (0..k).step_by(KC) {
            let kc = (pc + KC).min(k);
            let k_len = kc - pc;

            #[cfg(feature = "rayon")]
            {
                let c_addr = c.as_mut_ptr() as usize;
                let ic_indices: Vec<usize> = (0..m).step_by(MC).collect();

                ic_indices.into_par_iter().for_each(|ic| {
                    let c_ptr = c_addr as *mut f64;
                    let mc = (ic + MC).min(m);

                    for i in (ic..mc).step_by(MR) {
                        for j in (jc..nc).step_by(NR) {
                            let i_end = (i + MR).min(mc);
                            let j_end = (j + NR).min(nc);

                            if i_end == i + MR && j_end == j + NR {
                                if pc == 0 {
                                    for ii in i..i_end {
                                        for jj in j..j_end {
                                            *c_ptr.add(ii * n + jj) = 0.0;
                                        }
                                    }
                                }
                                kernel_4x2_f64_blocked(
                                    a,
                                    b,
                                    core::slice::from_raw_parts_mut(c_ptr, c.len()),
                                    i,
                                    j,
                                    pc,
                                    k_len,
                                    k,
                                    n,
                                );
                            } else {
                                for ii in i..i_end {
                                    for jj in j..j_end {
                                        if pc == 0 {
                                            *c_ptr.add(ii * n + jj) = 0.0;
                                        }
                                        for kk in pc..kc {
                                            *c_ptr.add(ii * n + jj) += a[ii * k + kk] * b[kk * n + jj];
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }

            #[cfg(not(feature = "rayon"))]
            {
                for ic in (0..m).step_by(MC) {
                    let mc = (ic + MC).min(m);

                    for i in (ic..mc).step_by(MR) {
                        for j in (jc..nc).step_by(NR) {
                            let i_end = (i + MR).min(mc);
                            let j_end = (j + NR).min(nc);

                            if i_end == i + MR && j_end == j + NR {
                                if pc == 0 {
                                    for ii in i..i_end {
                                        for jj in j..j_end {
                                            c[ii * n + jj] = 0.0;
                                        }
                                    }
                                }
                                kernel_4x2_f64_blocked(a, b, c, i, j, pc, k_len, k, n);
                            } else {
                                for ii in i..i_end {
                                    for jj in j..j_end {
                                        if pc == 0 {
                                            c[ii * n + jj] = 0.0;
                                        }
                                        for kk in pc..kc {
                                            c[ii * n + jj] += a[ii * k + kk] * b[kk * n + jj];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub unsafe fn f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    const MC: usize = 256;
    const KC: usize = 256;
    const NC: usize = 2048;
    const MR: usize = 4;
    const NR: usize = 2;

    let use_packing = m >= 512 && n >= 512 && k >= 512;

    if !use_packing {
        return f64_blocked_no_pack(a, b, c, m, k, n);
    }

    for jc in (0..n).step_by(NC) {
        let nc = (jc + NC).min(n) - jc;

        for pc in (0..k).step_by(KC) {
            let kc_val = (pc + KC).min(k) - pc;

            let packed_b_size = kc_val * ((nc + NR - 1) / NR) * NR;
            let mut packed_b = vec![0.0f64; packed_b_size];

            pack_b_f64(b, &mut packed_b, n, pc, jc, kc_val, nc);

            #[cfg(feature = "rayon")]
            {
                let c_addr = c.as_mut_ptr() as usize;
                let c_len = c.len();
                let ic_indices: Vec<usize> = (0..m).step_by(MC).collect();

                ic_indices.into_par_iter().for_each(|ic| {
                    let c_ptr = c_addr as *mut f64;
                    let mc = (ic + MC).min(m) - ic;
                    let packed_a_size = ((mc + MR - 1) / MR) * MR * kc_val;
                    let mut packed_a = vec![0.0f64; packed_a_size];

                    pack_a_f64(a, &mut packed_a, k, ic, pc, mc, kc_val);

                    for i in (0..mc).step_by(MR) {
                        for j in (0..nc).step_by(NR) {
                            let i_end = (i + MR).min(mc);
                            let j_end = (j + NR).min(nc);

                            if i_end == i + MR && j_end == j + NR {
                                if pc == 0 {
                                    for ii in 0..MR {
                                        for jj in 0..NR {
                                            *c_ptr.add((ic + i + ii) * n + (jc + j + jj)) = 0.0;
                                        }
                                    }
                                }
                                kernel_4x2_f64_packed(
                                    &packed_a,
                                    &packed_b,
                                    core::slice::from_raw_parts_mut(c_ptr, c_len),
                                    ic + i,
                                    jc + j,
                                    i,
                                    j,
                                    kc_val,
                                    n,
                                );
                            } else {
                                for ii in i..i_end {
                                    for jj in j..j_end {
                                        if pc == 0 {
                                            *c_ptr.add((ic + ii) * n + (jc + jj)) = 0.0;
                                        }
                                        for kk in 0..kc_val {
                                            *c_ptr.add((ic + ii) * n + (jc + jj)) +=
                                                a[(ic + ii) * k + pc + kk] * b[(pc + kk) * n + jc + jj];
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }

            #[cfg(not(feature = "rayon"))]
            {
                for ic in (0..m).step_by(MC) {
                    let mc = (ic + MC).min(m) - ic;
                    let packed_a_size = ((mc + MR - 1) / MR) * MR * kc_val;
                    let mut packed_a = vec![0.0f64; packed_a_size];

                    pack_a_f64(a, &mut packed_a, k, ic, pc, mc, kc_val);

                    for i in (0..mc).step_by(MR) {
                        for j in (0..nc).step_by(NR) {
                            let i_end = (i + MR).min(mc);
                            let j_end = (j + NR).min(nc);

                            if i_end == i + MR && j_end == j + NR {
                                if pc == 0 {
                                    for ii in 0..MR {
                                        for jj in 0..NR {
                                            c[(ic + i + ii) * n + (jc + j + jj)] = 0.0;
                                        }
                                    }
                                }
                                kernel_4x2_f64_packed(&packed_a, &packed_b, c, ic + i, jc + j, i, j, kc_val, n);
                            } else {
                                for ii in i..i_end {
                                    for jj in j..j_end {
                                        if pc == 0 {
                                            c[(ic + ii) * n + (jc + jj)] = 0.0;
                                        }
                                        for kk in 0..kc_val {
                                            c[(ic + ii) * n + (jc + jj)] +=
                                                a[(ic + ii) * k + pc + kk] * b[(pc + kk) * n + jc + jj];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
