use super::VjpCompute;
use crate::ops::{CmpOp, CmpScalarOp};

// CmpOp and CmpScalarOp are not differentiable (return boolean tensors)
impl VjpCompute for CmpOp {}
impl VjpCompute for CmpScalarOp {}
