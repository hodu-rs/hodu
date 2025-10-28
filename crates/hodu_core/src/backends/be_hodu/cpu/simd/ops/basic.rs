use crate::compat::*;

// Macro to generate SIMD binary operations for a specific type
macro_rules! impl_simd_binary {
    ($name:ident, $ty:ty, $simd_op:ident, $scalar_op:expr) => {
        pub fn $name(lhs: &[$ty], rhs: &[$ty]) -> Vec<$ty> {
            let len = lhs.len().min(rhs.len());
            let mut result: Vec<$ty> = Vec::with_capacity(len);

            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            #[cfg(target_feature = "neon")]
            {
                use crate::backends::be_hodu::cpu::simd::SimdType;
                const LANES: usize = <$ty>::LANES;

                unsafe {
                    let simd_len = len / LANES * LANES;

                    // SIMD processing
                    for i in (0..simd_len).step_by(LANES) {
                        let a = <$ty>::simd_load(lhs.as_ptr().add(i));
                        let b = <$ty>::simd_load(rhs.as_ptr().add(i));
                        let c = <$ty>::$simd_op(a, b);
                        <$ty>::simd_store(result.as_mut_ptr().add(i), c);
                    }

                    result.set_len(simd_len);
                }

                // Scalar tail
                for i in result.len()..len {
                    result.push($scalar_op(lhs[i], rhs[i]));
                }

                return result;
            }

            #[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), target_feature = "neon")))]
            {
                // Scalar fallback
                for i in 0..len {
                    result.push($scalar_op(lhs[i], rhs[i]));
                }
                result
            }
        }
    };
}

// f32 - 4 lanes
impl_simd_binary!(add_f32, f32, simd_add, |a, b| a + b);
impl_simd_binary!(sub_f32, f32, simd_sub, |a, b| a - b);
impl_simd_binary!(mul_f32, f32, simd_mul, |a, b| a * b);
impl_simd_binary!(div_f32, f32, simd_div, |a, b| a / b);

// f64 - 2 lanes
impl_simd_binary!(add_f64, f64, simd_add, |a, b| a + b);
impl_simd_binary!(sub_f64, f64, simd_sub, |a, b| a - b);
impl_simd_binary!(mul_f64, f64, simd_mul, |a, b| a * b);
impl_simd_binary!(div_f64, f64, simd_div, |a, b| a / b);

// u8 - 16 lanes
impl_simd_binary!(add_u8, u8, simd_add, |a, b| a + b);
impl_simd_binary!(sub_u8, u8, simd_sub, |a, b| a - b);
impl_simd_binary!(mul_u8, u8, simd_mul, |a, b| a * b);
impl_simd_binary!(div_u8, u8, simd_div, |a, b| a / b);

// u16 - 8 lanes
impl_simd_binary!(add_u16, u16, simd_add, |a, b| a + b);
impl_simd_binary!(sub_u16, u16, simd_sub, |a, b| a - b);
impl_simd_binary!(mul_u16, u16, simd_mul, |a, b| a * b);
impl_simd_binary!(div_u16, u16, simd_div, |a, b| a / b);

// u32 - 4 lanes
impl_simd_binary!(add_u32, u32, simd_add, |a, b| a + b);
impl_simd_binary!(sub_u32, u32, simd_sub, |a, b| a - b);
impl_simd_binary!(mul_u32, u32, simd_mul, |a, b| a * b);
impl_simd_binary!(div_u32, u32, simd_div, |a, b| a / b);

// i8 - 16 lanes
impl_simd_binary!(add_i8, i8, simd_add, |a, b| a + b);
impl_simd_binary!(sub_i8, i8, simd_sub, |a, b| a - b);
impl_simd_binary!(mul_i8, i8, simd_mul, |a, b| a * b);
impl_simd_binary!(div_i8, i8, simd_div, |a, b| a / b);

// i16 - 8 lanes
impl_simd_binary!(add_i16, i16, simd_add, |a, b| a + b);
impl_simd_binary!(sub_i16, i16, simd_sub, |a, b| a - b);
impl_simd_binary!(mul_i16, i16, simd_mul, |a, b| a * b);
impl_simd_binary!(div_i16, i16, simd_div, |a, b| a / b);

// i32 - 4 lanes
impl_simd_binary!(add_i32, i32, simd_add, |a, b| a + b);
impl_simd_binary!(sub_i32, i32, simd_sub, |a, b| a - b);
impl_simd_binary!(mul_i32, i32, simd_mul, |a, b| a * b);
impl_simd_binary!(div_i32, i32, simd_div, |a, b| a / b);

#[cfg(test)]
mod tests {
    use super::*;

    // f32 tests
    #[test]
    fn test_f32_add() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = add_f32(&a, &b);
        assert_eq!(result, vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_f32_sub() {
        let a = vec![5.0f32, 6.0, 7.0, 8.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = sub_f32(&a, &b);
        assert_eq!(result, vec![4.0f32, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_f32_mul() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![2.0f32, 2.0, 2.0, 2.0];
        let result = mul_f32(&a, &b);
        assert_eq!(result, vec![2.0f32, 4.0, 6.0, 8.0]);
    }

    // f64 tests
    #[test]
    fn test_f64_add() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        let b = vec![1.0f64, 1.0, 1.0, 1.0];
        let result = add_f64(&a, &b);
        assert_eq!(result, vec![2.0f64, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_f64_mul() {
        let a = vec![1.0f64, 2.0];
        let b = vec![3.0f64, 4.0];
        let result = mul_f64(&a, &b);
        assert_eq!(result, vec![3.0f64, 8.0]);
    }

    // u8 tests
    #[test]
    fn test_u8_add() {
        let a = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let b = vec![1u8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let result = add_u8(&a, &b);
        assert_eq!(result, vec![2u8, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]);
    }

    #[test]
    fn test_u8_mul() {
        let a = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let b = vec![2u8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        let result = mul_u8(&a, &b);
        assert_eq!(
            result,
            vec![2u8, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
        );
    }

    // u16 tests
    #[test]
    fn test_u16_add() {
        let a = vec![1u16, 2, 3, 4, 5, 6, 7, 8];
        let b = vec![1u16, 1, 1, 1, 1, 1, 1, 1];
        let result = add_u16(&a, &b);
        assert_eq!(result, vec![2u16, 3, 4, 5, 6, 7, 8, 9]);
    }

    // u32 tests
    #[test]
    fn test_u32_add() {
        let a = vec![1u32, 2, 3, 4];
        let b = vec![1u32, 1, 1, 1];
        let result = add_u32(&a, &b);
        assert_eq!(result, vec![2u32, 3, 4, 5]);
    }

    // i8 tests
    #[test]
    fn test_i8_add() {
        let a = vec![1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let b = vec![-1i8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
        let result = add_i8(&a, &b);
        assert_eq!(result, vec![0i8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    }

    // i16 tests
    #[test]
    fn test_i16_sub() {
        let a = vec![10i16, 20, 30, 40, 50, 60, 70, 80];
        let b = vec![1i16, 2, 3, 4, 5, 6, 7, 8];
        let result = sub_i16(&a, &b);
        assert_eq!(result, vec![9i16, 18, 27, 36, 45, 54, 63, 72]);
    }

    // i32 tests
    #[test]
    fn test_i32_mul() {
        let a = vec![1i32, 2, 3, 4];
        let b = vec![-2i32, -2, -2, -2];
        let result = mul_i32(&a, &b);
        assert_eq!(result, vec![-2i32, -4, -6, -8]);
    }
}
