use hodu_core::{error::HoduResult, tensor::Tensor};
pub use hodu_nn_macros::Optimizer;

pub trait Optimizer {
    fn step(&mut self, parameters: &mut [&mut Tensor]) -> HoduResult<()>;
    fn zero_grad(&mut self, parameters: &mut [&mut Tensor]) -> HoduResult<()>;
}
