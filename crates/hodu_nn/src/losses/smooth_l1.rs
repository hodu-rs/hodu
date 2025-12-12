use crate::module::Module;
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor};

/// Smooth L1 Loss (also known as Huber Loss with beta=1.0 by default).
///
/// Creates a criterion that uses a squared term if the absolute
/// element-wise error falls below beta and an L1 term otherwise.
///
/// loss(x, y) = 0.5 * (x - y)^2 / beta,  if |x - y| < beta
///            = |x - y| - 0.5 * beta,    otherwise
///
/// This is less sensitive to outliers than MSELoss and in some cases
/// prevents exploding gradients.
#[derive(Module, Clone)]
#[module(inputs = 2)]
pub struct SmoothL1Loss {
    beta: Scalar,
}

impl Default for SmoothL1Loss {
    fn default() -> Self {
        Self { beta: Scalar::F32(1.0) }
    }
}

impl SmoothL1Loss {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_beta(beta: impl Into<Scalar>) -> Self {
        Self { beta: beta.into() }
    }

    fn forward(&self, (pred, target): (&Tensor, &Tensor)) -> HoduResult<Tensor> {
        let dtype = pred.dtype();
        let diff = pred.sub(target)?;
        let abs_diff = diff.abs()?;

        let beta = self.beta.to_dtype(dtype);
        let half = Scalar::from_f32(0.5, dtype);
        let half_beta = beta * half;

        // Quadratic part: 0.5 * (x - y)^2 / beta
        let quadratic = diff.square()?.mul_scalar(half)?.div_scalar(beta)?;

        // Linear part: |x - y| - 0.5 * beta
        let linear = abs_diff.sub_scalar(half_beta)?;

        // Select based on condition: |x - y| < beta
        let mask = abs_diff.lt_scalar(beta)?;
        let loss = mask.mul(&quadratic)?.add(&mask.logical_not()?.mul(&linear)?)?;

        loss.mean_all()
    }
}
