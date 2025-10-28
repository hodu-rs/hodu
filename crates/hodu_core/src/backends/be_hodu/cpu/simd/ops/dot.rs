use crate::compat::*;

pub fn f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    super::matmul::f32(a, b, m, k, n)
}

pub fn f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    super::matmul::f64(a, b, m, k, n)
}

pub fn u8(a: &[u8], b: &[u8], m: usize, k: usize, n: usize) -> Vec<u8> {
    super::matmul::u8(a, b, m, k, n)
}

pub fn u16(a: &[u16], b: &[u16], m: usize, k: usize, n: usize) -> Vec<u16> {
    super::matmul::u16(a, b, m, k, n)
}

pub fn u32(a: &[u32], b: &[u32], m: usize, k: usize, n: usize) -> Vec<u32> {
    super::matmul::u32(a, b, m, k, n)
}

pub fn i8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i8> {
    super::matmul::i8(a, b, m, k, n)
}

pub fn i16(a: &[i16], b: &[i16], m: usize, k: usize, n: usize) -> Vec<i16> {
    super::matmul::i16(a, b, m, k, n)
}

pub fn i32(a: &[i32], b: &[i32], m: usize, k: usize, n: usize) -> Vec<i32> {
    super::matmul::i32(a, b, m, k, n)
}
