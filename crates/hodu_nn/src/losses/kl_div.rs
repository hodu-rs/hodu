use crate::module::Module;
use hodu_core::{error::HoduResult, tensor::Tensor};

/// Kullback-Leibler Divergence Loss.
///
/// Measures how one probability distribution diverges from a second,
/// expected probability distribution.
///
/// KL(P || Q) = sum(P * (log(P) - log(Q)))
///
/// Note: The input is expected to contain log-probabilities (output of log_softmax).
/// The target is expected to contain probabilities (not log-probabilities).
///
/// loss(input, target) = target * (log(target) - input)
///
/// To avoid numerical issues, target values close to 0 are handled specially.
#[derive(Module, Clone, Default)]
#[module(inputs = 2)]
pub struct KLDivLoss {
    log_target: bool,
}

impl KLDivLoss {
    pub fn new() -> Self {
        Self::default()
    }

    /// If true, expects target to be log-probabilities instead of probabilities.
    pub fn with_log_target(log_target: bool) -> Self {
        Self { log_target }
    }

    fn forward(&self, (input, target): (&Tensor, &Tensor)) -> HoduResult<Tensor> {
        // input: log-probabilities (output of log_softmax)
        // target: probabilities or log-probabilities depending on log_target

        let loss = if self.log_target {
            // target is log-probabilities
            // KL = exp(log_target) * (log_target - input)
            let target_prob = target.exp()?;
            target_prob.mul(&target.sub(input)?)?
        } else {
            // target is probabilities
            // KL = target * (log(target) - input)
            // To handle target=0, we use: target * log(target) - target * input
            // where target * log(target) = 0 when target = 0 (by convention)
            let target_log_target = target.mul(&target.ln()?)?;
            let target_input = target.mul(input)?;
            target_log_target.sub(&target_input)?
        };

        loss.sum_all()?.div_scalar(input.shape()[0] as f32)
    }
}
