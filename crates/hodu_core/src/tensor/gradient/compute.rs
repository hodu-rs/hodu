use super::{
    core::{get_active_context, is_computing_gradients, is_in_optimizer_step, set_computing_gradients, ContextId},
    tape::{get_tape_clone, push_entry, TapeEntry},
    vjp::VjpCompute,
};
use crate::{
    error::{HoduError, HoduResult},
    ops::{Op, OpParams},
    tensor::{self, set_grad_tensor_id, tensor_from_id, Tensor, TensorId},
};
use std::collections::HashMap;

pub fn record_operation(input_ids: Vec<TensorId>, output_id: TensorId, op: Op, op_params: OpParams) -> HoduResult<()> {
    if is_computing_gradients() {
        return Ok(());
    }

    if is_in_optimizer_step() {
        return Ok(());
    }

    let any_requires_grad = input_ids.iter().any(|&id| {
        use crate::tensor;
        tensor::with_tensor(id, |t| t.requires_grad).unwrap_or(false)
    });

    if !any_requires_grad {
        return Ok(());
    }

    let context_id = get_active_context();

    let entry = TapeEntry {
        output_id,
        op,
        op_params,
        input_ids,
    };

    push_entry(context_id, entry)
}

struct GradientComputationGuard {
    keep_alive: HashMap<TensorId, Tensor>,
}

impl GradientComputationGuard {
    fn new_with_capacity(capacity: usize) -> Self {
        set_computing_gradients(true);
        let keep_alive = HashMap::with_capacity(capacity);
        Self { keep_alive }
    }

    fn keep(&mut self, tensor: Tensor) {
        self.keep_alive.insert(tensor.id(), tensor);
    }
}

impl Drop for GradientComputationGuard {
    fn drop(&mut self) {
        set_computing_gradients(false);
    }
}

pub fn compute_gradients(loss_tensor_id: TensorId) -> HoduResult<()> {
    let loss_tensor = tensor_from_id(loss_tensor_id);
    let loss_shape = loss_tensor.shape();

    let loss_grad = Tensor::ones(&loss_shape, loss_tensor.dtype())?;
    let loss_grad_id = loss_grad.id();

    set_grad_tensor_id(loss_tensor_id, loss_grad_id)?;

    let context_id = get_active_context();
    let tape = get_tape_clone(context_id)?;

    let tape_len = tape.len();
    let mut guard = GradientComputationGuard::new_with_capacity(tape_len * 3);
    let mut gradients: HashMap<TensorId, TensorId> = HashMap::new();

    gradients.insert(loss_tensor_id, loss_grad_id);
    guard.keep(loss_tensor);
    guard.keep(loss_grad);

    let result = (|| -> HoduResult<()> {
        for entry in tape.iter().rev() {
            if let Some(&grad_output) = gradients.get(&entry.output_id) {
                for &input_id in &entry.input_ids {
                    if tensor::get(input_id).is_none() {
                        return Err(HoduError::TensorNotFound(input_id));
                    }
                }

                if tensor::get(grad_output).is_none() {
                    return Err(HoduError::TensorNotFound(grad_output));
                }

                guard.keep(tensor_from_id(grad_output));

                let input_grads = compute_vjp_for_op(
                    &entry.op,
                    &entry.op_params,
                    &entry.input_ids,
                    entry.output_id,
                    grad_output,
                )?;

                for &grad_id in &input_grads {
                    guard.keep(tensor_from_id(grad_id));
                }

                for (input_id, grad_id) in entry.input_ids.iter().zip(input_grads.iter()) {
                    if tensor::get(*grad_id).is_none() {
                        return Err(HoduError::TensorNotFound(*grad_id));
                    }

                    let final_grad_id = if let Some(&existing_grad) = gradients.get(input_id) {
                        if tensor::get(existing_grad).is_none() {
                            return Err(HoduError::TensorNotFound(existing_grad));
                        }
                        let result_id = super::vjp_utils::create_add_tensor(existing_grad, *grad_id)?;
                        guard.keep(tensor_from_id(result_id));
                        result_id
                    } else {
                        *grad_id
                    };

                    gradients.insert(*input_id, final_grad_id);

                    let input_tensor = tensor_from_id(*input_id);
                    if input_tensor.is_requires_grad() {
                        set_grad_tensor_id(*input_id, final_grad_id)?;
                    }
                    guard.keep(input_tensor);
                }
            }
        }

        Ok(())
    })();

    drop(guard);

    if result.is_ok() {
        use super::tape;

        if context_id == ContextId::DEFAULT {
            tape::cleanup_default_context_after_backward();
        }

        tape::clear_tape_for_context(context_id);
    }

    result
}

fn compute_vjp_for_op(
    op: &Op,
    op_params: &OpParams,
    inputs: &[TensorId],
    output: TensorId,
    grad_output: TensorId,
) -> HoduResult<Vec<TensorId>> {
    match op {
        Op::Binary(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        Op::BinaryLogical(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        Op::Cmp(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        Op::CmpScalar(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        Op::Unary(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        Op::UnaryLogical(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        Op::UnaryScalar(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        Op::Matrix(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        Op::Reduce(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        Op::Concat(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        Op::Split(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        Op::Indexing(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        Op::Conv(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        Op::Windowing(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        Op::Shape(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        Op::ShapeScalars(op) => op.compute_vjp(inputs, output, grad_output, op_params),
        _ => Err(HoduError::VjpFunctionNotFound(format!("compute_vjp for {:?}", op))),
    }
}
