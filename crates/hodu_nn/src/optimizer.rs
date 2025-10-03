use hodu_core::{error::HoduResult, scalar::Scalar, tensor::Tensor};
pub use hodu_nn_macros::Optimizer;

pub trait Optimizer {
    fn step(&mut self, parameters: &[&Tensor]) -> HoduResult<()>;
    fn zero_grad(&self, parameters: &[&Tensor]) -> HoduResult<()>;
    fn set_learning_rate(&mut self, learning_rate: impl Into<Scalar>);
}
