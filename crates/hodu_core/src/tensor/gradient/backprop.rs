//! Gradient computation (backward pass)
//!
//! This module implements reverse-mode automatic differentiation.

use super::{
    context::{get_active_context, ContextId},
    tape::{get_tape_clone, push_entry, TapeEntry},
    vjp::VjpCompute,
};
use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{Op, OpParams},
    scalar::Scalar,
    tensor::{self, set_grad_tensor_id, tensor_from_id, Tensor, TensorId},
};

/// Flag indicating whether we're currently computing gradients
static IS_COMPUTING_GRADIENTS: AtomicBool = AtomicBool::new(false);

/// Flag indicating whether we're currently in optimizer step
/// Optimizer operations should not be recorded on the tape
static IS_IN_OPTIMIZER_STEP: AtomicBool = AtomicBool::new(false);

/// Check if gradients are currently being computed
pub fn is_computing_gradients() -> bool {
    IS_COMPUTING_GRADIENTS.load(Ordering::Relaxed)
}

/// Check if we're in optimizer step
pub fn is_in_optimizer_step() -> bool {
    IS_IN_OPTIMIZER_STEP.load(Ordering::Relaxed)
}

/// Set optimizer step flag
pub fn set_optimizer_step_flag(value: bool) {
    IS_IN_OPTIMIZER_STEP.store(value, Ordering::Relaxed);
}

/// Record an operation on the gradient tape (no parameters)
pub fn record_operation(output_id: TensorId, op: Op, input_ids: Vec<TensorId>) -> HoduResult<()> {
    record_operation_with_params(output_id, op, input_ids, OpParams::default())
}

/// Record an operation with a scalar parameter
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

/// Record an operation with scalar parameters
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

/// Record an operation with dimension parameters
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

/// Record a split operation with output index
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

/// Record an operation on the gradient tape with full parameters (internal)
///
/// Called by tensor operations to record the computational graph.
/// Only records if NOT currently computing gradients (to avoid infinite recursion).
/// Also skips recording if no tensor in the operation requires grad.
pub(crate) fn record_operation_with_params(
    output_id: TensorId,
    op: Op,
    input_ids: Vec<TensorId>,
    op_params: OpParams,
) -> HoduResult<()> {
    // Don't record during gradient computation to avoid infinite recursion
    if is_computing_gradients() {
        return Ok(());
    }

    // Don't record during optimizer step
    if is_in_optimizer_step() {
        return Ok(());
    }

    // Don't record if no input requires grad
    let any_requires_grad = input_ids.iter().any(|&id| {
        use crate::tensor;
        tensor::with_tensor(id, |t| t.requires_grad).unwrap_or(false)
    });

    if !any_requires_grad {
        return Ok(());
    }

    // Get current context
    let context_id = get_active_context();

    // Create tape entry
    let entry = TapeEntry {
        output_id,
        op,
        op_params,
        input_ids,
    };

    // Add to tape
    push_entry(context_id, entry)
}

/// RAII guard to manage gradient computation lifecycle
///
/// Tensors stored in this guard are kept alive during gradient computation.
/// When dropped, IS_COMPUTING_GRADIENTS flag is set to false.
struct GradientComputationGuard {
    keep_alive: HashMap<TensorId, Tensor>,
}

impl GradientComputationGuard {
    fn new_with_capacity(capacity: usize) -> Self {
        IS_COMPUTING_GRADIENTS.store(true, Ordering::Relaxed);
        Self {
            keep_alive: HashMap::with_capacity(capacity),
        }
    }

    fn keep(&mut self, tensor: Tensor) {
        self.keep_alive.insert(tensor.id(), tensor);
    }
}

impl Drop for GradientComputationGuard {
    fn drop(&mut self) {
        // Set flag to false BEFORE dropping tensors
        IS_COMPUTING_GRADIENTS.store(false, Ordering::Relaxed);
        // Now self.keep_alive will be dropped
    }
}

/// Compute gradients for all tensors in the computational graph
///
/// Performs reverse-mode automatic differentiation starting from the loss tensor.
pub fn compute_gradients(loss_tensor_id: TensorId) -> HoduResult<()> {
    let loss_tensor = tensor_from_id(loss_tensor_id);
    let loss_shape = loss_tensor.shape();

    let loss_grad = Tensor::ones(&loss_shape, loss_tensor.dtype())?;
    let loss_grad_id = loss_grad.id();

    set_grad_tensor_id(loss_tensor_id, loss_grad_id)?;

    // Get tape from active context
    let context_id = get_active_context();
    let tape = get_tape_clone(context_id)?;

    // Pre-allocate based on tape size for better performance
    let tape_len = tape.len();
    let mut guard = GradientComputationGuard::new_with_capacity(tape_len * 3);
    let mut gradients: HashMap<TensorId, TensorId> = HashMap::new();

    gradients.insert(loss_tensor_id, loss_grad_id);
    guard.keep(loss_tensor);
    guard.keep(loss_grad);

    let result = (|| -> HoduResult<()> {
        for entry in tape.iter().rev() {
            if let Some(&grad_output) = gradients.get(&entry.output_id) {
                // Verify all input tensors exist
                for &input_id in &entry.input_ids {
                    if tensor::get(input_id).is_none() {
                        return Err(HoduError::TensorNotFound(input_id));
                    }
                }

                if tensor::get(grad_output).is_none() {
                    return Err(HoduError::TensorNotFound(grad_output));
                }

                guard.keep(tensor_from_id(grad_output));

                // Compute VJP for this operation
                let input_grads = compute_vjp_for_op(
                    &entry.op,
                    &entry.op_params,
                    &entry.input_ids,
                    entry.output_id,
                    grad_output,
                )?;

                // Keep all computed gradients alive during backward pass
                // VJP functions return TensorId after dropping the Tensor handle,
                // so we must keep them alive here
                for &grad_id in &input_grads {
                    guard.keep(tensor_from_id(grad_id));
                }

                // Accumulate gradients for each input
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

    // Explicitly drop guard to set IS_COMPUTING_GRADIENTS = false
    // and clean up keep_alive Vec before returning
    drop(guard);

    if result.is_ok() {
        use super::tape;

        // Clean up DEFAULT context's intermediate tensors with ref_count=0
        // Must do this BEFORE clearing tape (cleanup needs tape to find tensors)
        if context_id == ContextId::DEFAULT {
            tape::cleanup_default_context_after_backward();
        }

        // Clear the tape to avoid accumulating entries across iterations
        tape::clear_tape_for_context(context_id);
    }

    result
}

/// Dispatch VJP computation to the appropriate operation
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
