use super::VjpCompute;
use crate::{
    error::{HoduError, HoduResult},
    op_params::{EinsumParams, OpParams},
    ops::EinsumOp,
    tensor::{tensor_from_id, TensorId},
};

impl VjpCompute for EinsumOp {
    fn compute_vjp(
        &self,
        inputs: &[TensorId],
        _output: TensorId,
        grad_output: TensorId,
        op_params: &OpParams,
    ) -> HoduResult<Vec<TensorId>> {
        match self {
            EinsumOp::Einsum => {
                let OpParams::Einsum(EinsumParams {
                    input_subscripts,
                    output_subscripts,
                    ..
                }) = op_params
                else {
                    return Err(HoduError::VjpFunctionNotFound(
                        "Einsum requires EinsumParams".to_string(),
                    ));
                };

                let grad_tensor = tensor_from_id(grad_output);
                let mut grad_inputs = Vec::with_capacity(inputs.len());

                for (i, _input_id) in inputs.iter().enumerate() {
                    // Build gradient equation for input i
                    // grad_input_i = einsum(grad_equation, [grad, other_inputs...])
                    //
                    // The grad equation:
                    // - First operand: grad_output with output_subscripts
                    // - Other operands: all inputs except input i with their subscripts
                    // - Result: input_subscripts[i]

                    let mut grad_operand_subs = Vec::new();
                    let mut grad_operand_tensors = Vec::new();

                    // First operand is the gradient output
                    grad_operand_subs.push(output_subscripts.iter().collect::<String>());
                    grad_operand_tensors.push(&grad_tensor);

                    // Remaining operands are inputs except the one we're computing gradient for
                    let input_tensors: Vec<_> = inputs.iter().map(|&id| tensor_from_id(id)).collect();
                    for (j, input_tensor) in input_tensors.iter().enumerate() {
                        if j != i {
                            grad_operand_subs.push(input_subscripts[j].iter().collect::<String>());
                            grad_operand_tensors.push(input_tensor);
                        }
                    }

                    // Result subscripts are the subscripts of the input we're computing gradient for
                    let result_subs: String = input_subscripts[i].iter().collect();

                    // Build the equation string
                    let grad_equation = format!("{}->{}", grad_operand_subs.join(","), result_subs);

                    // Compute the gradient
                    let grad_tensor_refs: Vec<&crate::tensor::Tensor> = grad_operand_tensors.to_vec();
                    let grad_input = crate::tensor::Tensor::einsum(&grad_equation, &grad_tensor_refs)?;

                    grad_inputs.push(grad_input.id());
                }

                Ok(grad_inputs)
            },
        }
    }
}
