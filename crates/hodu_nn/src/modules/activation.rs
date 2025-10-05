use crate::compat::*;
use crate::module::Module;
use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor};

#[derive(Module, Clone)]
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

#[derive(Module, Clone)]
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

#[derive(Module, Clone)]
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

#[derive(Module, Clone)]
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

#[derive(Module, Clone)]
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
