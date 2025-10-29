use crate::compat::*;
use crate::module::Module;
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor};

#[derive(Module, Clone, Default)]
#[module(inputs = 2)]
pub struct MAELoss;

impl MAELoss {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, (pred, target): (&Tensor, &Tensor)) -> HoduResult<Tensor> {
        let diff = pred.sub(target)?;
        let abs_diff = diff.abs()?;

        let pred_layout = pred.get_layout();
        let pred_shape = pred_layout.get_shape();
        let batch_size = Scalar::new(pred_shape[0]);
        let mean = abs_diff.sum_all()?.div_scalar(batch_size)?;

        Ok(mean)
    }
}
