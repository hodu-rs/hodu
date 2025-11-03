use super::utils::broadcast_tensors3;
use crate::{error::HoduResult, tensor::Tensor};

impl Tensor {
    pub fn where3_select(condition: &Tensor, x: &Tensor, y: &Tensor) -> HoduResult<Self> {
        let (condition, x, y) = broadcast_tensors3(condition, x, y)?;

        let mask = condition.to_dtype(x.get_dtype())?;
        let one = Self::ones_like(&mask)?;
        let inv_mask = one.sub(&mask)?;

        let x_part = mask.mul(&x)?;
        let y_part = inv_mask.mul(&y)?;
        x_part.add(&y_part)
    }
}
