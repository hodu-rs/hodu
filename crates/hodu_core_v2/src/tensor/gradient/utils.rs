use crate::{
    error::HoduResult,
    scalar::Scalar,
    tensor::{tensor_from_id, Tensor, TensorId},
    types::Shape,
};

// Binary operations
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

pub fn create_sub_tensor(a: TensorId, b: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    let tensor_b = tensor_from_id(b);
    tensor_a.sub(&tensor_b).map(|t| t.id())
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

// CMP operations
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

// CMP Scalar operations
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

pub fn create_sqrt_tensor(a: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    tensor_a.sqrt().map(|t| t.id())
}

// Unary Scalar operations
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

// Reduce operations
pub fn create_sum_to_shape_tensor(tensor_id: TensorId, target_shape: &Shape) -> HoduResult<TensorId> {
    let tensor = tensor_from_id(tensor_id);
    tensor.sum_to_shape(target_shape).map(|t| t.id())
}

// Utility functions
pub fn create_zeros_like_tensor(a: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    let shape = tensor_a.shape();
    let dtype = tensor_a.dtype();
    let zeros_tensor = Tensor::zeros(&shape, dtype)?;
    Ok(zeros_tensor.id())
}

pub fn create_ones_like_tensor(a: TensorId) -> HoduResult<TensorId> {
    let tensor_a = tensor_from_id(a);
    let shape = tensor_a.shape();
    let dtype = tensor_a.dtype();
    let ones_tensor = Tensor::ones(&shape, dtype)?;
    Ok(ones_tensor.id())
}
