use crate::{layer::compat::*, tensor::TensorId};

impl fmt::Display for TensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for TensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TensorId[{}]", self.0)
    }
}
