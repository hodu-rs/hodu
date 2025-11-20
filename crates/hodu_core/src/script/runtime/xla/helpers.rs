use crate::{
    error::{HoduError, HoduResult},
    script::{builder::ir::Attribute, op_params::OpParams},
};
use hodu_xla::{XlaBuilder, XlaOp};
use std::collections::HashMap;

/// Helper function for unary operations with input validation
pub fn unary_op_check<F>(inputs: &[XlaOp], f: F, op_name: &str) -> HoduResult<XlaOp>
where
    F: FnOnce(&[XlaOp]) -> Result<XlaOp, hodu_xla::Error>,
{
    if inputs.len() != 1 {
        return Err(HoduError::InternalError(format!("{} requires 1 input", op_name)));
    }
    Ok(f(inputs)?)
}

/// Helper function for binary operations with input validation
pub fn binary_op_check<F>(inputs: &[XlaOp], expected: usize, f: F, op_name: &str) -> HoduResult<XlaOp>
where
    F: FnOnce(&[XlaOp]) -> Result<XlaOp, hodu_xla::Error>,
{
    if inputs.len() != expected {
        return Err(HoduError::InternalError(format!(
            "{} requires {} inputs",
            op_name, expected
        )));
    }
    Ok(f(inputs)?)
}

/// Helper function for scalar operations
pub fn scalar_op<F>(
    builder: &XlaBuilder,
    inputs: &[XlaOp],
    attributes: &HashMap<String, Attribute>,
    f: F,
    op_name: &str,
) -> HoduResult<XlaOp>
where
    F: FnOnce(&[XlaOp], XlaOp) -> Result<XlaOp, hodu_xla::Error>,
{
    if inputs.len() != 1 {
        return Err(HoduError::InternalError(format!("{} requires 1 input", op_name)));
    }
    let scalar = get_scalar_from_attributes(attributes)?;
    let scalar_op = builder.constant_r0(scalar)?;
    Ok(f(inputs, scalar_op)?)
}

/// Extract scalar value from attributes
pub fn get_scalar_from_attributes(attributes: &HashMap<String, Attribute>) -> HoduResult<f32> {
    let params = OpParams::new(attributes);
    let scalar = params.get_scalar("scalar")?;
    Ok(scalar.to_f32())
}
