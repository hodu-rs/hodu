use crate::{error::HoduResult, scalar::Scalar, tensor::Tensor};

// Normalization Operations
impl Tensor {
    pub fn softmax<D: Into<Scalar>>(&self, dim: D) -> HoduResult<Self> {
        let dim_scalar = dim.into();
        let dim_usize = dim_scalar.to_u64() as usize;

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
        let dim_usize = dim_scalar.to_u64() as usize;

        // Numerical stability: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
        let max_val = self.max(&[dim_usize], true)?;
        let shifted = self.sub(&max_val)?;
        let exp_vals = shifted.exp()?;
        let sum_exp = exp_vals.sum(&[dim_usize], true)?;
        let log_sum_exp = sum_exp.ln()?;
        shifted.sub(&log_sum_exp)
    }
}
