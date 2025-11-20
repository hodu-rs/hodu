use hodu_core::{error::HoduResult, tensor::Tensor};
pub use hodu_nn_macros::Optimizer;

pub trait Optimizer {
    fn step(&mut self, parameters: &[&Tensor]) -> HoduResult<()>;
    fn zero_grad(&self, parameters: &[&Tensor]) -> HoduResult<()>;
}
