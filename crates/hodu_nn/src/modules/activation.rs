use crate::{
    compat::*,
    module::Module,
    state::{get_state, State},
};
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor};

#[derive(Module, Clone, Default)]
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        input.relu()
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Module, Clone, Default)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        input.sigmoid()
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Module, Clone, Default)]
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        input.tanh()
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Module, Clone, Default)]
pub struct Gelu;

impl Gelu {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        input.gelu()
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Module, Clone, Default)]
pub struct Softplus;

impl Softplus {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        input.softplus()
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Module, Clone, Default)]
pub struct SiLU;

impl SiLU {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        input.silu()
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Module, Clone, Default)]
pub struct Swish;

impl Swish {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        input.swish()
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Module, Clone, Default)]
pub struct Mish;

impl Mish {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        input.mish()
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Module, Clone)]
pub struct LeakyReLU {
    exponent: Scalar,
}

impl LeakyReLU {
    pub fn new(exponent: impl Into<Scalar>) -> Self {
        Self {
            exponent: exponent.into(),
        }
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let exponent = self.exponent.to_dtype(input.get_dtype());
        input.leaky_relu(exponent)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Module, Clone)]
pub struct ELU {
    exponent: Scalar,
}

impl ELU {
    pub fn new(exponent: impl Into<Scalar>) -> Self {
        Self {
            exponent: exponent.into(),
        }
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let exponent = self.exponent.to_dtype(input.get_dtype());
        input.elu(exponent)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Module, Clone)]
pub struct PReLU {
    weight: Scalar,
}

impl PReLU {
    pub fn new(weight: impl Into<Scalar>) -> Self {
        Self { weight: weight.into() }
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let weight = self.weight.to_dtype(input.get_dtype());
        input.prelu(weight)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[derive(Module, Clone)]
pub struct RReLU {
    lower: Scalar,
    upper: Scalar,
}

impl RReLU {
    pub fn new(lower: impl Into<Scalar>, upper: impl Into<Scalar>) -> Self {
        Self {
            lower: lower.into(),
            upper: upper.into(),
        }
    }

    pub fn forward(&self, input: &Tensor) -> HoduResult<Tensor> {
        let dtype = input.get_dtype();
        let zero = Scalar::zero(dtype);

        // Compute alpha based on training/evaluation mode
        let alpha = if get_state() == State::Training {
            // Training mode: use random alpha for each element
            let lower_f32 = self.lower.to_f32();
            let upper_f32 = self.upper.to_f32();
            Tensor::rand_uniform_like(input, lower_f32, upper_f32)?
        } else {
            // Evaluation mode: use average of lower and upper bounds
            let avg = (self.lower.to_f32() + self.upper.to_f32()) / 2.0;
            let avg_scalar = Scalar::from_f32(avg, dtype);
            Tensor::full_like(input, avg_scalar)?
        };

        // RReLU: x if x > 0, else alpha * x
        let mask_pos = input.gt_scalar(zero)?;
        let mask_neg = input.le_scalar(zero)?;
        let positive_part = input.mul(&mask_pos.to_dtype(dtype)?)?;
        let negative_part = input.mul(&alpha)?.mul(&mask_neg.to_dtype(dtype)?)?;

        positive_part.add(&negative_part)
    }

    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}
