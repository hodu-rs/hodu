use crate::optimizer::Optimizer;
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor};

/// RMSprop optimizer.
///
/// Implements the RMSprop algorithm proposed by Hinton.
///
/// v_t = α * v_{t-1} + (1 - α) * g_t²
/// θ_t = θ_{t-1} - lr * g_t / (√v_t + ε)
///
/// With momentum:
/// v_t = α * v_{t-1} + (1 - α) * g_t²
/// m_t = momentum * m_{t-1} + lr * g_t / (√v_t + ε)
/// θ_t = θ_{t-1} - m_t
#[derive(Optimizer, Clone)]
pub struct RMSprop {
    learning_rate: Scalar,
    alpha: Scalar,
    epsilon: Scalar,
    momentum: Scalar,
    weight_decay: Scalar,
    centered: bool,
    v: Vec<Tensor>,
    m: Vec<Tensor>,
    g_avg: Vec<Tensor>,
}

impl RMSprop {
    /// Creates a new RMSprop optimizer.
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate (default: 0.01)
    /// * `alpha` - Smoothing constant (default: 0.99)
    /// * `epsilon` - Term added for numerical stability (default: 1e-8)
    /// * `momentum` - Momentum factor (default: 0)
    /// * `weight_decay` - Weight decay (L2 penalty) (default: 0)
    /// * `centered` - If true, compute centered RMSprop (default: false)
    pub fn new(
        learning_rate: impl Into<Scalar>,
        alpha: impl Into<Scalar>,
        epsilon: impl Into<Scalar>,
        momentum: impl Into<Scalar>,
        weight_decay: impl Into<Scalar>,
        centered: bool,
    ) -> Self {
        Self {
            learning_rate: learning_rate.into(),
            alpha: alpha.into(),
            epsilon: epsilon.into(),
            momentum: momentum.into(),
            weight_decay: weight_decay.into(),
            centered,
            v: Vec::new(),
            m: Vec::new(),
            g_avg: Vec::new(),
        }
    }

    /// Creates RMSprop with default parameters.
    pub fn default_with_lr(learning_rate: impl Into<Scalar>) -> Self {
        Self::new(learning_rate, 0.99, 1e-8, 0.0, 0.0, false)
    }

    fn step(&mut self, parameters: &[&Tensor]) -> HoduResult<()> {
        // Initialize state on first call
        if self.v.is_empty() {
            self.v = parameters
                .iter()
                .map(|param| Tensor::zeros_like(param))
                .collect::<HoduResult<Vec<_>>>()?;

            if self.momentum.to_f32() != 0.0 {
                self.m = parameters
                    .iter()
                    .map(|param| Tensor::zeros_like(param))
                    .collect::<HoduResult<Vec<_>>>()?;
            }

            if self.centered {
                self.g_avg = parameters
                    .iter()
                    .map(|param| Tensor::zeros_like(param))
                    .collect::<HoduResult<Vec<_>>>()?;
            }
        }

        let use_momentum = self.momentum.to_f32() != 0.0;

        for (i, param) in parameters.iter().enumerate() {
            let mut grad = param.grad()?;

            let lr = self.learning_rate.to_dtype(grad.dtype());
            let alpha = self.alpha.to_dtype(grad.dtype());
            let epsilon = self.epsilon.to_dtype(grad.dtype());
            let weight_decay = self.weight_decay.to_dtype(grad.dtype());
            let one = Scalar::one(grad.dtype());

            // Apply weight decay
            if weight_decay.to_f32() != 0.0 {
                grad = grad.add(&param.mul_scalar(weight_decay)?)?;
            }

            let one_minus_alpha = one - alpha;

            // v_t = α * v_{t-1} + (1 - α) * g_t²
            self.v[i] = self.v[i]
                .mul_scalar(alpha)?
                .add(&grad.square()?.mul_scalar(one_minus_alpha)?)?;

            let avg = if self.centered {
                // g_avg_t = α * g_avg_{t-1} + (1 - α) * g_t
                self.g_avg[i] = self.g_avg[i]
                    .mul_scalar(alpha)?
                    .add(&grad.mul_scalar(one_minus_alpha)?)?;
                // v_t - g_avg_t²
                self.v[i].sub(&self.g_avg[i].square()?)?
            } else {
                self.v[i].clone()
            };

            if use_momentum {
                let momentum = self.momentum.to_dtype(grad.dtype());
                // m_t = momentum * m_{t-1} + g_t / (√avg + ε)
                self.m[i] = self.m[i]
                    .mul_scalar(momentum)?
                    .add(&grad.div(&avg.sqrt()?.add_scalar(epsilon)?)?)?;
                // θ_t = θ_{t-1} - lr * m_t
                param.set_(&param.sub(&self.m[i].mul_scalar(lr)?)?)?;
            } else {
                // θ_t = θ_{t-1} - lr * g_t / (√avg + ε)
                let update = grad.div(&avg.sqrt()?.add_scalar(epsilon)?)?;
                param.set_(&param.sub(&update.mul_scalar(lr)?)?)?;
            }
        }

        Ok(())
    }

    pub fn set_learning_rate(&mut self, learning_rate: impl Into<Scalar>) {
        self.learning_rate = learning_rate.into();
    }
}
