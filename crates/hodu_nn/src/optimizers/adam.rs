use crate::compat::*;
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

    fn step(&mut self, parameters: &mut [&mut Tensor]) -> HoduResult<()> {
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

        for ((param, m), v) in parameters.iter_mut().zip(self.m.iter_mut()).zip(self.v.iter_mut()) {
            let grad = param.grad()?;
            let lr = self.learning_rate.to_dtype(grad.dtype());
            let beta1 = self.beta1.to_dtype(grad.dtype());
            let beta2 = self.beta2.to_dtype(grad.dtype());
            let epsilon = self.epsilon.to_dtype(grad.dtype());
            let one = Scalar::one(grad.dtype());

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
            param.set_(&param.sub(&step.mul_scalar(lr_t)?)?)?;
        }

        Ok(())
    }

    pub fn set_learning_rate(&mut self, learning_rate: impl Into<Scalar>) {
        self.learning_rate = learning_rate.into();
    }
}

#[derive(Optimizer, Clone)]
pub struct AdamW {
    learning_rate: Scalar,
    beta1: Scalar,
    beta2: Scalar,
    epsilon: Scalar,
    weight_decay: Scalar,
    t: usize,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl AdamW {
    pub fn new(
        learning_rate: impl Into<Scalar>,
        beta1: impl Into<Scalar>,
        beta2: impl Into<Scalar>,
        epsilon: impl Into<Scalar>,
        weight_decay: impl Into<Scalar>,
    ) -> Self {
        Self {
            learning_rate: learning_rate.into(),
            beta1: beta1.into(),
            beta2: beta2.into(),
            epsilon: epsilon.into(),
            weight_decay: weight_decay.into(),
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    fn step(&mut self, parameters: &mut [&mut Tensor]) -> HoduResult<()> {
        // Initialize momentum buffers on first call
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

        for ((param, m), v) in parameters.iter_mut().zip(self.m.iter_mut()).zip(self.v.iter_mut()) {
            let grad = param.grad()?;

            // Convert scalars to match gradient dtype
            let lr = self.learning_rate.to_dtype(grad.dtype());
            let beta1 = self.beta1.to_dtype(grad.dtype());
            let beta2 = self.beta2.to_dtype(grad.dtype());
            let epsilon = self.epsilon.to_dtype(grad.dtype());
            let weight_decay = self.weight_decay.to_dtype(grad.dtype());
            let one = Scalar::one(grad.dtype());

            // Compute bias correction terms: β₁^t and β₂^t
            let mut beta1_t = beta1;
            let mut beta2_t = beta2;
            for _ in 1..self.t {
                beta1_t = beta1_t * beta1;
                beta2_t = beta2_t * beta2;
            }

            let one_minus_beta1 = one - beta1;
            let one_minus_beta2 = one - beta2;
            let one_minus_beta1_t = one - beta1_t;
            let one_minus_beta2_t = one - beta2_t;

            // Update biased first moment estimate: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
            *m = m.mul_scalar(beta1)?.add(&grad.mul_scalar(one_minus_beta1)?)?;

            // Update biased second raw moment estimate: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
            *v = v.mul_scalar(beta2)?.add(&grad.square()?.mul_scalar(one_minus_beta2)?)?;

            // Compute bias-corrected first moment: m̂_t = m_t / (1 - β₁^t)
            let m_hat = m.div_scalar(one_minus_beta1_t)?;

            // Compute bias-corrected second moment: v̂_t = v_t / (1 - β₂^t)
            let v_hat = v.div_scalar(one_minus_beta2_t)?;

            // Compute adaptive learning rate step: lr * m̂_t / (√v̂_t + ε)
            let adaptive_step = m_hat.div(&v_hat.sqrt()?.add_scalar(epsilon)?)?;

            // AdamW: Decoupled weight decay
            // First apply weight decay: θ = θ * (1 - lr * weight_decay)
            let decay_factor = one - lr * weight_decay;
            let decayed_param = param.mul_scalar(decay_factor)?;

            // Then apply adaptive gradient update: θ = θ - lr * m̂ / (√v̂ + ε)
            param.set_(&decayed_param.sub(&adaptive_step.mul_scalar(lr)?)?)?;
        }

        Ok(())
    }

    pub fn set_learning_rate(&mut self, learning_rate: impl Into<Scalar>) {
        self.learning_rate = learning_rate.into();
    }

    pub fn set_weight_decay(&mut self, weight_decay: impl Into<Scalar>) {
        self.weight_decay = weight_decay.into();
    }

    pub fn get_learning_rate(&self) -> Scalar {
        self.learning_rate
    }

    pub fn get_weight_decay(&self) -> Scalar {
        self.weight_decay
    }
}
