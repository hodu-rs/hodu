use crate::{
    error::HoduResult,
    scalar::Scalar,
    tensor,
    tensor::{tensor_from_id, TensorId},
};

// Basic binary operations
pub fn create_add_tensor(a: TensorId, b: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    let tensor_b = tensor_from_id(b);
    tensor_a.add(&tensor_b).map(|t| t.id())
}

pub fn create_mul_tensor(a: TensorId, b: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    let tensor_b = tensor_from_id(b);
    tensor_a.mul(&tensor_b).map(|t| t.id())
}

pub fn create_div_tensor(a: TensorId, b: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    let tensor_b = tensor_from_id(b);
    tensor_a.div(&tensor_b).map(|t| t.id())
}

pub fn create_pow_tensor(a: TensorId, b: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    let tensor_b = tensor_from_id(b);
    tensor_a.pow(&tensor_b).map(|t| t.id())
}

// Unary operations
pub fn create_neg_tensor(a: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.neg().map(|t| t.id())
}

pub fn create_sign_tensor(a: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.sign().map(|t| t.id())
}

pub fn create_recip_tensor(a: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.recip().map(|t| t.id())
}

pub fn create_sigmoid_tensor(a: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.sigmoid().map(|t| t.id())
}

pub fn create_tanh_tensor(a: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.tanh().map(|t| t.id())
}

pub fn create_sin_tensor(a: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.sin().map(|t| t.id())
}

pub fn create_cos_tensor(a: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.cos().map(|t| t.id())
}

pub fn create_ln_tensor(a: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.ln().map(|t| t.id())
}

pub fn create_exp_tensor(a: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.exp().map(|t| t.id())
}

// Scalar operations
pub fn create_add_scalar_tensor(a: TensorId, scalar: Scalar) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.add_scalar(scalar).map(|t| t.id())
}

pub fn create_sub_scalar_tensor(a: TensorId, scalar: Scalar) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.sub_scalar(scalar).map(|t| t.id())
}

pub fn create_mul_scalar_tensor(a: TensorId, scalar: Scalar) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.mul_scalar(scalar).map(|t| t.id())
}

pub fn create_pow_scalar_tensor(a: TensorId, scalar: Scalar) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.pow_scalar(scalar).map(|t| t.id())
}

// Comparison operations that return masks (as tensors)
pub fn create_gt_scalar_tensor(a: TensorId, scalar: Scalar) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.gt_scalar(scalar).map(|t| t.id())
}

pub fn create_le_scalar_tensor(a: TensorId, scalar: Scalar) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.le_scalar(scalar).map(|t| t.id())
}

pub fn create_ge_scalar_tensor(a: TensorId, scalar: Scalar) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.ge_scalar(scalar).map(|t| t.id())
}

// Binary comparison operations
pub fn create_ge_tensor(a: TensorId, b: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    let tensor_b = tensor_from_id(b);
    tensor_a.ge(&tensor_b).map(|t| t.id())
}

pub fn create_gt_tensor(a: TensorId, b: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    let tensor_b = tensor_from_id(b);
    tensor_a.gt(&tensor_b).map(|t| t.id())
}

pub fn create_le_tensor(a: TensorId, b: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    let tensor_b = tensor_from_id(b);
    tensor_a.le(&tensor_b).map(|t| t.id())
}

pub fn create_lt_tensor(a: TensorId, b: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    let tensor_b = tensor_from_id(b);
    tensor_a.lt(&tensor_b).map(|t| t.id())
}

// Helper function to create scalar with matching dtype
pub fn create_scalar_for_dtype(value: f64, dtype: crate::types::dtype::DType) -> Scalar {
    match dtype {
        crate::types::dtype::DType::F64 => Scalar::F64(value),
        crate::types::dtype::DType::F32 => Scalar::F32(value as f32),
        crate::types::dtype::DType::F16 => Scalar::F16(half::f16::from_f64(value)),
        crate::types::dtype::DType::BF16 => Scalar::BF16(half::bf16::from_f64(value)),
        crate::types::dtype::DType::F8E5M2 => Scalar::F8E5M2(float8::F8E5M2::from(value as f32)),
        crate::types::dtype::DType::F8E4M3 => Scalar::F8E4M3(float8::F8E4M3::from(value as f32)),
        _ => Scalar::F32(value as f32), // fallback for non-float types
    }
}

// Utility functions
pub fn create_zeros_like_tensor(a: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    let layout = tensor_a.get_layout();
    let dtype = tensor_a.get_dtype();
    let zeros_tensor = tensor::Tensor::zeros(layout.get_shape(), dtype)?;
    Ok(zeros_tensor.id())
}
