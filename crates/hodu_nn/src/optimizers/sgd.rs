use crate::optimizer::Optimizer;
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor};

#[derive(Optimizer, Clone)]
pub struct SGD {
    learning_rate: Scalar,
}

impl SGD {
    pub fn new(learning_rate: impl Into<Scalar>) -> Self {
        Self {
            learning_rate: learning_rate.into(),
        }
    }

    pub fn step(&mut self, parameters: &[&Tensor]) -> HoduResult<()> {
        for param in parameters.iter() {
            let grad = param.grad()?;
            let lr = self.learning_rate.to_dtype(grad.get_dtype());

            param.set(&param.sub(&grad.mul_scalar(lr)?)?)?;
        }
        Ok(())
    }

    pub fn zero_grad(&self, parameters: &[&Tensor]) -> HoduResult<()> {
        for param in parameters.iter() {
            param.zero_grad()?;
        }
        Ok(())
    }

    pub fn set_learning_rate(&mut self, learning_rate: impl Into<Scalar>) {
        self.learning_rate = learning_rate.into();
    }
}
