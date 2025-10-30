use crate::compat::*;

// ============================================================================
// NEON-optimized versions
// ============================================================================
#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))]
pub fn f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        super::super::neon::matmul::f32(a, b, &mut c, m, k, n);
    }
    c
}

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))]
pub fn f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    unsafe {
        super::super::neon::matmul::f64(a, b, &mut c, m, k, n);
    }
    c
}

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))]
pub fn u8(a: &[u8], b: &[u8], m: usize, k: usize, n: usize) -> Vec<u8> {
    let mut c = vec![0u8; m * n];
    unsafe {
        super::super::neon::matmul::u8(a, b, &mut c, m, k, n);
    }
    c
}

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))]
pub fn u16(a: &[u16], b: &[u16], m: usize, k: usize, n: usize) -> Vec<u16> {
    let mut c = vec![0u16; m * n];
    unsafe {
        super::super::neon::matmul::u16(a, b, &mut c, m, k, n);
    }
    c
}

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))]
pub fn u32(a: &[u32], b: &[u32], m: usize, k: usize, n: usize) -> Vec<u32> {
    let mut c = vec![0u32; m * n];
    unsafe {
        super::super::neon::matmul::u32(a, b, &mut c, m, k, n);
    }
    c
}

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))]
pub fn i8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i8> {
    let mut c = vec![0i8; m * n];
    unsafe {
        super::super::neon::matmul::i8(a, b, &mut c, m, k, n);
    }
    c
}

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))]
pub fn i16(a: &[i16], b: &[i16], m: usize, k: usize, n: usize) -> Vec<i16> {
    let mut c = vec![0i16; m * n];
    unsafe {
        super::super::neon::matmul::i16(a, b, &mut c, m, k, n);
    }
    c
}

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon"))]
pub fn i32(a: &[i32], b: &[i32], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    unsafe {
        super::super::neon::matmul::i32(a, b, &mut c, m, k, n);
    }
    c
}

// ============================================================================
// Scalar fallback versions
// ============================================================================
#[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon")))]
pub fn f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
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

#[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon")))]
pub fn f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
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

#[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon")))]
pub fn u8(a: &[u8], b: &[u8], m: usize, k: usize, n: usize) -> Vec<u8> {
    let mut c = vec![0u8; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0u32;
            for kk in 0..k {
                sum += a[i * k + kk] as u32 * b[kk * n + j] as u32;
            }
            c[i * n + j] = sum.min(255) as u8;
        }
    }
    c
}

#[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon")))]
pub fn u16(a: &[u16], b: &[u16], m: usize, k: usize, n: usize) -> Vec<u16> {
    let mut c = vec![0u16; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0u64;
            for kk in 0..k {
                sum += a[i * k + kk] as u64 * b[kk * n + j] as u64;
            }
            c[i * n + j] = sum.min(65535) as u16;
        }
    }
    c
}

#[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon")))]
pub fn u32(a: &[u32], b: &[u32], m: usize, k: usize, n: usize) -> Vec<u32> {
    let mut c = vec![0u32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0u64;
            for kk in 0..k {
                sum += a[i * k + kk] as u64 * b[kk * n + j] as u64;
            }
            c[i * n + j] = sum.min(u32::MAX as u64) as u32;
        }
    }
    c
}

#[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon")))]
pub fn i8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i8> {
    let mut c = vec![0i8; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i32;
            for kk in 0..k {
                sum += a[i * k + kk] as i32 * b[kk * n + j] as i32;
            }
            c[i * n + j] = sum.clamp(-128, 127) as i8;
        }
    }
    c
}

#[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon")))]
pub fn i16(a: &[i16], b: &[i16], m: usize, k: usize, n: usize) -> Vec<i16> {
    let mut c = vec![0i16; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i64;
            for kk in 0..k {
                sum += a[i * k + kk] as i64 * b[kk * n + j] as i64;
            }
            c[i * n + j] = sum.clamp(-32768, 32767) as i16;
        }
    }
    c
}

#[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon")))]
pub fn i32(a: &[i32], b: &[i32], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i64;
            for kk in 0..k {
                sum += a[i * k + kk] as i64 * b[kk * n + j] as i64;
            }
            c[i * n + j] = sum.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }
    c
}
