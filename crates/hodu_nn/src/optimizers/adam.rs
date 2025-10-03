use crate::optimizer::Optimizer;
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor};

#[derive(Optimizer, Clone)]
pub struct Adam {
    learning_rate: Scalar,
    beta1: Scalar,
    beta2: Scalar,
    epsilon: Scalar,
    t: usize,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl Adam {
    pub fn new(
        learning_rate: impl Into<Scalar>,
        beta1: impl Into<Scalar>,
        beta2: impl Into<Scalar>,
        epsilon: impl Into<Scalar>,
    ) -> Self {
        Self {
            learning_rate: learning_rate.into(),
            beta1: beta1.into(),
            beta2: beta2.into(),
            epsilon: epsilon.into(),
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn step(&mut self, parameters: &[&Tensor]) -> HoduResult<()> {
        if self.m.is_empty() || self.v.is_empty() {
            self.m = parameters
                .iter()
                .map(|param| Tensor::zeros_like(param))
                .collect::<HoduResult<Vec<_>>>()?;
            self.v = parameters
                .iter()
                .map(|param| Tensor::zeros_like(param))
                .collect::<HoduResult<Vec<_>>>()?;
        }

        self.t += 1;

        for ((param, m), v) in parameters.iter().zip(self.m.iter_mut()).zip(self.v.iter_mut()) {
            let grad = param.grad()?;
            let lr = self.learning_rate.to_dtype(grad.get_dtype());
            let beta1 = self.beta1.to_dtype(grad.get_dtype());
            let beta2 = self.beta2.to_dtype(grad.get_dtype());
            let epsilon = self.epsilon.to_dtype(grad.get_dtype());
            let one = Scalar::one(grad.get_dtype());

            let mut beta1_t = beta1;
            let mut beta2_t = beta2;
            for _ in 0..self.t {
                beta1_t = beta1_t * beta1;
                beta2_t = beta2_t * beta2;
            }

            let one_minus_beta1_t = one - beta1_t;
            let one_minus_beta2_t = one - beta2_t;
            let lr_t = lr * (one_minus_beta2_t.sqrt()) / one_minus_beta1_t;

            let one_minus_beta1 = one - beta1;
            let one_minus_beta2 = one - beta2;

            // Update biased first moment estimate
            *m = m.mul_scalar(beta1)?.add(&grad.mul_scalar(one_minus_beta1)?)?;

            // Update biased second raw moment estimate
            *v = v.mul_scalar(beta2)?.add(&grad.square()?.mul_scalar(one_minus_beta2)?)?;

            // Compute bias-corrected estimates
            let m_hat = m.div_scalar(one_minus_beta1_t)?;
            let v_hat = v.div_scalar(one_minus_beta2_t)?;

            // Update parameters
            let step = m_hat.div(&v_hat.sqrt()?.add_scalar(epsilon)?)?;
            param.set(&param.sub(&step.mul_scalar(lr_t)?)?)?;
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
