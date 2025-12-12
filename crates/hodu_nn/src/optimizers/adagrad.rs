use crate::optimizer::Optimizer;
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor};

/// Adagrad optimizer.
///
/// Implements the Adagrad algorithm.
///
/// v_t = v_{t-1} + g_t²
/// θ_t = θ_{t-1} - lr * g_t / (√v_t + ε)
///
/// Adagrad adapts the learning rate for each parameter based on
/// the historical sum of squared gradients.
#[derive(Optimizer, Clone)]
pub struct Adagrad {
    learning_rate: Scalar,
    lr_decay: Scalar,
    epsilon: Scalar,
    weight_decay: Scalar,
    initial_accumulator_value: Scalar,
    t: usize,
    v: Vec<Tensor>,
}

impl Adagrad {
    /// Creates a new Adagrad optimizer.
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate (default: 0.01)
    /// * `lr_decay` - Learning rate decay (default: 0)
    /// * `epsilon` - Term added for numerical stability (default: 1e-10)
    /// * `weight_decay` - Weight decay (L2 penalty) (default: 0)
    /// * `initial_accumulator_value` - Initial value for the accumulator (default: 0)
    pub fn new(
        learning_rate: impl Into<Scalar>,
        lr_decay: impl Into<Scalar>,
        epsilon: impl Into<Scalar>,
        weight_decay: impl Into<Scalar>,
        initial_accumulator_value: impl Into<Scalar>,
    ) -> Self {
        Self {
            learning_rate: learning_rate.into(),
            lr_decay: lr_decay.into(),
            epsilon: epsilon.into(),
            weight_decay: weight_decay.into(),
            initial_accumulator_value: initial_accumulator_value.into(),
            t: 0,
            v: Vec::new(),
        }
    }

    /// Creates Adagrad with default parameters.
    pub fn default_with_lr(learning_rate: impl Into<Scalar>) -> Self {
        Self::new(learning_rate, 0.0, 1e-10, 0.0, 0.0)
    }

    fn step(&mut self, parameters: &[&Tensor]) -> HoduResult<()> {
        // Initialize state on first call
        if self.v.is_empty() {
            let init_val = self.initial_accumulator_value.to_f32();
            if init_val == 0.0 {
                self.v = parameters
                    .iter()
                    .map(|param| Tensor::zeros_like(param))
                    .collect::<HoduResult<Vec<_>>>()?;
            } else {
                self.v = parameters
                    .iter()
                    .map(|param| Tensor::full_like(param, self.initial_accumulator_value))
                    .collect::<HoduResult<Vec<_>>>()?;
            }
        }

        self.t += 1;

        for (i, param) in parameters.iter().enumerate() {
            let mut grad = param.grad()?;

            let lr = self.learning_rate.to_dtype(grad.dtype());
            let lr_decay = self.lr_decay.to_dtype(grad.dtype());
            let epsilon = self.epsilon.to_dtype(grad.dtype());
            let weight_decay = self.weight_decay.to_dtype(grad.dtype());
            let one = Scalar::one(grad.dtype());
            let t_scalar = Scalar::from_usize(self.t, grad.dtype());

            // Apply weight decay
            if weight_decay.to_f32() != 0.0 {
                grad = grad.add(&param.mul_scalar(weight_decay)?)?;
            }

            // Compute decayed learning rate
            // lr_t = lr / (1 + (t - 1) * lr_decay)
            let t_minus_one = t_scalar - one;
            let lr_t = lr / (one + t_minus_one * lr_decay);

            // v_t = v_{t-1} + g_t²
            self.v[i] = self.v[i].add(&grad.square()?)?;

            // θ_t = θ_{t-1} - lr_t * g_t / (√v_t + ε)
            let update = grad.div(&self.v[i].sqrt()?.add_scalar(epsilon)?)?;
            param.set_(&param.sub(&update.mul_scalar(lr_t)?)?)?;
        }

        Ok(())
    }

    pub fn set_learning_rate(&mut self, learning_rate: impl Into<Scalar>) {
        self.learning_rate = learning_rate.into();
    }
}
