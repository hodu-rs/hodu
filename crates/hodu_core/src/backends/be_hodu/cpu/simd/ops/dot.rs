use crate::compat::*;

pub fn f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    super::matmul::f32(a, b, m, k, n)
}

pub fn f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    super::matmul::f64(a, b, m, k, n)
}
