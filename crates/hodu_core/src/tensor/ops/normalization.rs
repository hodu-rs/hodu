use crate::{error::HoduResult, scalar::Scalar, tensor::Tensor};

impl Tensor {
    pub fn softmax<T: Into<Scalar>>(&self, dim: T) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as u32
        } else {
            dim_i32 as u32
        } as usize;

        // Numerical stability: subtract max
        // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        let max_val = self.max(&[dim_usize], true)?;
        let shifted = self.sub(&max_val)?;
        let exp_vals = shifted.exp()?;
        let sum_exp = exp_vals.sum(&[dim_usize], true)?;
        exp_vals.div(&sum_exp)
    }

    pub fn log_softmax<D: Into<Scalar>>(&self, dim: D) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_i32 = dim_scalar.to_i32();
        let ndim = self.ndim() as i32;
        let dim_usize = if dim_i32 < 0 {
            (ndim + dim_i32) as u32
        } else {
            dim_i32 as u32
        } as usize;

        // Numerical stability: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
        let max_val = self.max(&[dim_usize], true)?;
        let shifted = self.sub(&max_val)?;
        let exp_vals = shifted.exp()?;
        let sum_exp = exp_vals.sum(&[dim_usize], true)?;
        let log_sum_exp = sum_exp.ln()?;
        shifted.sub(&log_sum_exp)
    }
}
