use crate::module::Module;
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor};

#[derive(Module, Clone)]
#[module(inputs = 2)]
pub struct HuberLoss {
    delta: Scalar,
}

impl HuberLoss {
    pub fn new(delta: impl Into<Scalar>) -> Self {
        Self { delta: delta.into() }
    }

    fn forward(&self, (pred, target): (&Tensor, &Tensor)) -> HoduResult<Tensor> {
        let diff = pred.sub(target)?;
        let abs_diff = diff.abs()?;

        let quadratic_loss = diff
            .pow_scalar(Scalar::from_f32(2.0, diff.dtype()))?
            .div_scalar(Scalar::from_f32(2.0, diff.dtype()))?;
        let linear_loss = abs_diff
            .mul_scalar(self.delta)?
            .sub_scalar(self.delta.powi(2) / Scalar::from_f32(2.0, self.delta.dtype()))?;

        let mask = abs_diff.le_scalar(self.delta)?;

        let loss = mask
            .mul(&quadratic_loss)?
            .add(&mask.logical_not()?.mul(&linear_loss)?)?;

        let pred_shape = pred.shape();
        let batch_size = Scalar::from_f32(pred_shape[0] as f32, diff.dtype());
        let mean = loss.sum_all()?.div_scalar(batch_size)?;

        Ok(mean)
    }
}
