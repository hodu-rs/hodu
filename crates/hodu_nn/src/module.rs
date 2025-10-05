use crate::compat::*;
use hodu_core::{error::HoduResult, tensor::Tensor};
pub use hodu_nn_macros::Module;

pub trait Module<I = &'static Tensor> {
    fn forward(&self, input: I) -> HoduResult<Tensor>;
    fn parameters(&mut self) -> Vec<&mut Tensor>;
}
