use crate::compat::*;
use crate::module::Module;
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor};

#[derive(Module, Clone)]
#[module(inputs = 2)]
pub struct BCELoss {
    epsilon: Scalar,
}

impl Default for BCELoss {
    fn default() -> Self {
        Self {
            epsilon: Scalar::F32(1e-7), // for numerical stability
        }
    }
}

impl BCELoss {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_epsilon(epsilon: impl Into<Scalar>) -> Self {
        Self {
            epsilon: epsilon.into(),
        }
    }

    fn forward(&self, (pred, target): (&Tensor, &Tensor)) -> HoduResult<Tensor> {
        // Clamp predictions to avoid log(0)
        // pred_clamped = clamp(pred, epsilon, 1 - epsilon)
        let one_minus_eps_scalar = Scalar::one(self.epsilon.dtype()) - self.epsilon;
        let pred_clamped = pred.clamp(self.epsilon, one_minus_eps_scalar)?;

        // BCE = -[target * log(pred) + (1 - target) * log(1 - pred)]

        // First term: target * log(pred)
        let log_pred = pred_clamped.ln()?;
        let first_term = target.mul(&log_pred)?;

        // Second term: (1 - target) * log(1 - pred)
        let one = Tensor::ones_like(target)?;
        let one_minus_target = one.sub(target)?;
        let one_minus_pred = Tensor::ones_like(&pred_clamped)?.sub(&pred_clamped)?;
        let log_one_minus_pred = one_minus_pred.ln()?;
        let second_term = one_minus_target.mul(&log_one_minus_pred)?;

        // Combine
        let bce = first_term.add(&second_term)?.neg()?;

        // Mean over all elements
        bce.mean_all()
    }
}

#[derive(Module, Clone, Default)]
#[module(inputs = 2)]
pub struct BCEWithLogitsLoss;

impl BCEWithLogitsLoss {
    pub fn new() -> Self {
        Self
    }

    fn forward(&self, (logits, target): (&Tensor, &Tensor)) -> HoduResult<Tensor> {
        // More numerically stable: BCE(sigmoid(x)) = log(1 + exp(-x)) if target=1
        //                                            = log(1 + exp(x))  if target=0
        // This can be simplified to: max(x, 0) - x * target + log(1 + exp(-|x|))

        let zeros = Tensor::zeros_like(logits)?;
        let max_val = logits.maximum(&zeros)?; // max(x, 0)

        let neg_abs = logits.abs()?.neg()?; // -|x|
        let log_term = neg_abs.exp()?.add_scalar(Scalar::one(neg_abs.dtype()))?.ln()?; // log(1 + exp(-|x|))

        let target_term = logits.mul(target)?; // x * target

        // max(x, 0) - x * target + log(1 + exp(-|x|))
        let loss = max_val.sub(&target_term)?.add(&log_term)?;

        // Mean over all elements
        loss.mean_all()
    }
}
