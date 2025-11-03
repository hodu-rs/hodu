use crate::compat::*;
use crate::module::Module;
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor};

#[derive(Module, Clone, Default)]
#[module(inputs = 2)]
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, (pred, target): (&Tensor, &Tensor)) -> HoduResult<Tensor> {
        let diff = pred.sub(target)?;
        let squared = diff.pow_scalar(Scalar::from_f32(2.0, diff.dtype()))?;

        let squared_size = squared.layout().size();
        let num_elements = Scalar::new(squared_size).to_dtype(squared.dtype());
        let mean = squared.sum_all()?.div_scalar(num_elements)?;

        Ok(mean)
    }
}
