use crate::compat::*;
use crate::module::Module;
use crate::state::{get_state, State};
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor};

#[derive(Module, Clone)]
pub struct Dropout {
    p: Scalar,
}

impl Dropout {
    pub fn new(p: impl Into<Scalar>) -> Self {
        Dropout { p: p.into() }
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        // Only apply dropout in training mode
        if get_state() == State::Evaluation {
            return Ok(input.clone());
        }

        // Training mode: apply dropout
        let random = Tensor::rand_uniform_like(input, 0.0, 1.0)?;
        let mask = random.gt_scalar(self.p)?.to_dtype(input.dtype())?;
        let scale = 1.0 / (1.0 - self.p.to_f32());
        let scaled_mask = mask.mul_scalar(scale)?;

        input.mul(&scaled_mask)
    }
}
