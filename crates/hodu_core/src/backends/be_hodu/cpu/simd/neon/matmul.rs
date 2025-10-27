#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use core::arch::arm::*;

pub unsafe fn f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    const MR: usize = 4;
    const NR: usize = 4;

    for i in (0..m).step_by(MR) {
        for j in (0..n).step_by(NR) {
            let i_end = (i + MR).min(m);
            let j_end = (j + NR).min(n);

            if i_end == i + MR && j_end == j + NR {
                kernel_4x4_f32(a, b, c, i, j, k, n);
            } else {
                for ii in i..i_end {
                    for jj in j..j_end {
                        let mut sum = 0.0f32;
                        for kk in 0..k {
                            sum += a[ii * k + kk] * b[kk * n + jj];
                        }
                        c[ii * n + jj] = sum;
                    }
                }
            }
        }
    }
}

pub unsafe fn f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    const MR: usize = 2;
    const NR: usize = 2;

    for i in (0..m).step_by(MR) {
        for j in (0..n).step_by(NR) {
            let i_end = (i + MR).min(m);
            let j_end = (j + NR).min(n);

            if i_end == i + MR && j_end == j + NR {
                kernel_2x2_f64(a, b, c, i, j, k, n);
            } else {
                for ii in i..i_end {
                    for jj in j..j_end {
                        let mut sum = 0.0f64;
                        for kk in 0..k {
                            sum += a[ii * k + kk] * b[kk * n + jj];
                        }
                        c[ii * n + jj] = sum;
                    }
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn kernel_4x4_f32(a: &[f32], b: &[f32], c: &mut [f32], i: usize, j: usize, k: usize, n: usize) {
    let mut c0 = vdupq_n_f32(0.0);
    let mut c1 = vdupq_n_f32(0.0);
    let mut c2 = vdupq_n_f32(0.0);
    let mut c3 = vdupq_n_f32(0.0);

    for kk in 0..k {
        let a0 = *a.get_unchecked(i * k + kk);
        let a1 = *a.get_unchecked((i + 1) * k + kk);
        let a2 = *a.get_unchecked((i + 2) * k + kk);
        let a3 = *a.get_unchecked((i + 3) * k + kk);

        let b_vec = vld1q_f32(b.as_ptr().add(kk * n + j));

        c0 = vfmaq_n_f32(c0, b_vec, a0);
        c1 = vfmaq_n_f32(c1, b_vec, a1);
        c2 = vfmaq_n_f32(c2, b_vec, a2);
        c3 = vfmaq_n_f32(c3, b_vec, a3);
    }

    vst1q_f32(c.as_mut_ptr().add(i * n + j), c0);
    vst1q_f32(c.as_mut_ptr().add((i + 1) * n + j), c1);
    vst1q_f32(c.as_mut_ptr().add((i + 2) * n + j), c2);
    vst1q_f32(c.as_mut_ptr().add((i + 3) * n + j), c3);
}

#[inline(always)]
unsafe fn kernel_2x2_f64(a: &[f64], b: &[f64], c: &mut [f64], i: usize, j: usize, k: usize, n: usize) {
    let mut c0 = vdupq_n_f64(0.0);
    let mut c1 = vdupq_n_f64(0.0);

    for kk in 0..k {
        let a0 = *a.get_unchecked(i * k + kk);
        let a1 = *a.get_unchecked((i + 1) * k + kk);

        let b_vec = vld1q_f64(b.as_ptr().add(kk * n + j));

        c0 = vfmaq_n_f64(c0, b_vec, a0);
        c1 = vfmaq_n_f64(c1, b_vec, a1);
    }

    vst1q_f64(c.as_mut_ptr().add(i * n + j), c0);
    vst1q_f64(c.as_mut_ptr().add((i + 1) * n + j), c1);
}
