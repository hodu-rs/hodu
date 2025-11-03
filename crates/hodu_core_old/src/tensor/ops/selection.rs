use crate::{
    error::HoduResult,
    scalar::Scalar,
    tensor::{utils::broadcast_tensors3, Tensor},
};

// Selection Operations
impl Tensor {
    pub fn where3(&self, condition: &Tensor, other: &Tensor) -> HoduResult<Self> {
        let (condition, x, y) = broadcast_tensors3(condition, self, other)?;

        let mask = condition.to_dtype(x.get_dtype())?;
        let one = Self::ones_like(&mask)?;
        let inv_mask = one.sub(&mask)?;

        let x_part = mask.mul(&x)?;
        let y_part = inv_mask.mul(&y)?;
        x_part.add(&y_part)
    }

    pub fn masked_fill<T: Into<Scalar>>(&self, mask: &Tensor, value: T) -> HoduResult<Self> {
        let value_scalar = value.into();
        let fill_tensor = Self::full_like(self, value_scalar)?;
        self.where3(mask, &fill_tensor)
    }

    pub fn clamp<T: Into<Scalar>>(&self, min: T, max: T) -> HoduResult<Self> {
        let min_scalar = min.into();
        let max_scalar = max.into();

        let min_tensor = Self::full_like(self, min_scalar)?;
        let clamped_min = self.where3(&self.lt_scalar(min_scalar)?, &min_tensor)?;

        let max_tensor = Self::full_like(&clamped_min, max_scalar)?;
        clamped_min.where3(&clamped_min.gt_scalar(max_scalar)?, &max_tensor)
    }

    pub fn clamp_min<T: Into<Scalar>>(&self, min: T) -> HoduResult<Self> {
        let min_scalar = min.into();
        let min_tensor = Self::full_like(self, min_scalar)?;
        self.where3(&self.lt_scalar(min_scalar)?, &min_tensor)
    }

    pub fn clamp_max<T: Into<Scalar>>(&self, max: T) -> HoduResult<Self> {
        let max_scalar = max.into();
        let max_tensor = Self::full_like(self, max_scalar)?;
        self.where3(&self.gt_scalar(max_scalar)?, &max_tensor)
    }
}
