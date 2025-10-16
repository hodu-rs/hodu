use crate::compat::*;
use crate::module::Module;
use hodu_core::{error::HoduResult, tensor::Tensor};

#[derive(Module, Clone)]
#[module(inputs = 2)]
pub struct NLLLoss {
    dim: i32,
}

impl NLLLoss {
    pub fn new() -> Self {
        Self { dim: -1 }
    }

    pub fn with_dim(dim: i32) -> Self {
        Self { dim }
    }

    pub fn forward(&self, (log_probs, target): (&Tensor, &Tensor)) -> HoduResult<Tensor> {
        // Normalize dim to positive index
        let rank = log_probs.get_layout().get_shape().len() as i32;
        let gather_dim = if self.dim < 0 { rank + self.dim } else { self.dim };

        // Gather the log probabilities for the correct classes
        // target shape: [...] -> unsqueeze to [..., 1] at gather_dim
        let target_unsqueezed = target.unsqueeze(-1)?;

        // Gather along class dimension
        let gathered = log_probs.gather(gather_dim as i32, &target_unsqueezed)?;

        // Remove the extra dimension
        let gathered_squeezed = gathered.squeeze(Some(-1))?;

        // Negative log likelihood
        let nll = gathered_squeezed.neg()?;

        // Mean over batch
        nll.mean_all()
    }
}
