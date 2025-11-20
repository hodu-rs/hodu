use crate::{compat::*, tensor::Tensor};

/// Execution inputs - map of input names to tensors
pub type ExecutionInputs<'a> = HashMap<&'a str, Tensor>;

/// Execution outputs - map of output names to tensors
pub type ExecutionOutputs = HashMap<String, Tensor>;
