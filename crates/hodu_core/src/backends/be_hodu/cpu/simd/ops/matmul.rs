use crate::compat::*;

pub fn f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    #[cfg(target_feature = "neon")]
    {
        unsafe {
            super::super::neon::matmul::f32(a, b, &mut c, m, k, n);
        }
        return c;
    }

    #[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon")))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }
}

pub fn f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    #[cfg(target_feature = "neon")]
    {
        unsafe {
            super::super::neon::matmul::f64(a, b, &mut c, m, k, n);
        }
        return c;
    }

    #[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon")))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }
}
