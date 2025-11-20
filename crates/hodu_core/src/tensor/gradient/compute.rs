use super::{
    core::{get_active_context, is_computing_gradients, is_in_optimizer_step, set_computing_gradients, ContextId},
    tape::{get_tape_clone, push_entry, TapeEntry},
    vjp::VjpCompute,
};
use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    ops::{Op, OpParams},
    scalar::Scalar,
    tensor::{self, set_grad_tensor_id, tensor_from_id, Tensor, TensorId},
};

pub fn record_operation(output_id: TensorId, op: Op, input_ids: Vec<TensorId>) -> HoduResult<()> {
    record_operation_with_params(output_id, op, input_ids, OpParams::default())
}

pub fn record_operation_with_scalar(
    output_id: TensorId,
    op: Op,
    input_ids: Vec<TensorId>,
    scalar: Scalar,
) -> HoduResult<()> {
    record_operation_with_params(
        output_id,
        op,
        input_ids,
        OpParams {
            scalar: Some(scalar),
            ..Default::default()
        },
    )
}

pub fn record_operation_with_scalars(
    output_id: TensorId,
    op: Op,
    input_ids: Vec<TensorId>,
    scalars: Vec<Scalar>,
) -> HoduResult<()> {
    record_operation_with_params(
        output_id,
        op,
        input_ids,
        OpParams {
            scalars,
            ..Default::default()
        },
    )
}

pub fn record_operation_with_dims(
    output_id: TensorId,
    op: Op,
    input_ids: Vec<TensorId>,
    dims: Vec<Scalar>,
    keep_dim: Option<bool>,
) -> HoduResult<()> {
    record_operation_with_params(
        output_id,
        op,
        input_ids,
        OpParams {
            dims,
            keep_dim,
            ..Default::default()
        },
    )
}

pub fn record_operation_with_split_info(
    output_id: TensorId,
    op: Op,
    input_ids: Vec<TensorId>,
    params: Vec<Scalar>,
    output_index: usize,
) -> HoduResult<()> {
    record_operation_with_params(
        output_id,
        op,
        input_ids,
        OpParams {
            scalars: params,
            output_index: Some(output_index),
            ..Default::default()
        },
    )
}

pub(crate) fn record_operation_with_params(
    output_id: TensorId,
    op: Op,
    input_ids: Vec<TensorId>,
    op_params: OpParams,
) -> HoduResult<()> {
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
    fn new_with_capacity(_capacity: usize) -> Self {
        set_computing_gradients(true);
        #[cfg(feature = "std")]
        let keep_alive = HashMap::with_capacity(_capacity);
        #[cfg(not(feature = "std"))]
        let keep_alive = HashMap::new();
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
        Op::Binary(binary_op) => binary_op.compute_vjp(inputs, output, grad_output),
        Op::BinaryLogical(binary_logical_op) => binary_logical_op.compute_vjp(inputs, output, grad_output),
        Op::Cmp(cmp_op) => cmp_op.compute_vjp(inputs, output, grad_output),
        Op::CmpScalar(cmp_scalar_op) => {
            let scalar = op_params
                .scalar
                .ok_or_else(|| HoduError::VjpFunctionNotFound("CmpScalar requires scalar parameter".to_string()))?;
            cmp_scalar_op.compute_vjp_with_scalar(inputs, output, grad_output, scalar)
        },
        Op::Unary(unary_op) => unary_op.compute_vjp(inputs, output, grad_output),
        Op::UnaryLogical(unary_logical_op) => unary_logical_op.compute_vjp(inputs, output, grad_output),
        Op::UnaryScalar(unary_scalar_op) => {
            let scalar = op_params
                .scalar
                .ok_or_else(|| HoduError::VjpFunctionNotFound("UnaryScalar requires scalar parameter".to_string()))?;
            unary_scalar_op.compute_vjp_with_scalar(inputs, output, grad_output, scalar)
        },
        Op::Matrix(matrix_op) => matrix_op.compute_vjp(inputs, output, grad_output),
        Op::Reduce(reduce_op) => reduce_op.compute_vjp_with_dims(inputs, output, grad_output, &op_params.dims),
        Op::Concat(concat_op) => concat_op.compute_vjp_with_dims(inputs, output, grad_output, &op_params.scalars),
        Op::Split(split_op) => {
            let output_index = op_params
                .output_index
                .ok_or_else(|| HoduError::VjpFunctionNotFound("Split requires output_index parameter".to_string()))?;
            split_op.compute_vjp_with_split_info(inputs, output, grad_output, &op_params.scalars, output_index)
        },
        Op::Indexing(indexing_op) => indexing_op.compute_vjp_with_dims(inputs, output, grad_output, &op_params.dims),
        Op::Conv(conv_op) => conv_op.compute_vjp_with_scalars(inputs, output, grad_output, &op_params.scalars),
        Op::Windowing(windowing_op) => {
            windowing_op.compute_vjp_with_scalars(inputs, output, grad_output, &op_params.scalars)
        },
        Op::Shape(shape_op) => shape_op.compute_vjp(inputs, output, grad_output),
        Op::ShapeScalars(shape_op) => {
            shape_op.compute_vjp_with_scalars(inputs, output, grad_output, &op_params.scalars)
        },
        _ => Err(HoduError::VjpFunctionNotFound(format!("compute_vjp for {:?}", op))),
    }
}
