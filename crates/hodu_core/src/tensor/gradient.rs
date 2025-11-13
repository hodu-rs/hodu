mod binary;
mod cmp;
mod concat_split;
mod conv;
mod indexing;
mod matrix;
mod reduce;
mod shape;
mod unary;
mod utils;
mod windowing;

use crate::{
    error::{HoduError, HoduResult},
    layer::compat::*,
    ops::{Op, OpParams},
    scalar::Scalar,
    tensor::{self, set_grad_tensor_id, tensor_from_id, Tensor, TensorId},
};
use utils::create_add_tensor;

pub(crate) trait VjpCompute {
    fn compute_vjp(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(
            "compute_vjp not implemented".to_string(),
        ))
    }

    fn compute_vjp_with_scalar(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _scalar: Scalar,
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(
            "compute_vjp_with_scalar not implemented".to_string(),
        ))
    }

    fn compute_vjp_with_scalars(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _scalars: &[Scalar],
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(
            "compute_vjp_with_scalars not implemented".to_string(),
        ))
    }

    fn compute_vjp_with_dims(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _dims: &[Scalar],
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(
            "compute_vjp_with_dims not implemented".to_string(),
        ))
    }

    fn compute_vjp_with_split_info(
        &self,
        _inputs: &[TensorId],
        _output: TensorId,
        _grad_output: TensorId,
        _params: &[Scalar],
        _output_index: usize,
    ) -> HoduResult<Vec<TensorId>> {
        Err(HoduError::VjpFunctionNotFound(
            "compute_vjp_with_split_info not implemented".to_string(),
        ))
    }
}

#[derive(Clone)]
struct TapeEntry {
    output_id: TensorId,
    op: Op,
    op_params: OpParams,
    input_ids: Vec<TensorId>,
    #[allow(dead_code)]
    input_tensors: Vec<Tensor>, // Keep input tensors alive until backward completes
    #[allow(dead_code)]
    output_tensor: Tensor, // Keep output tensor alive until backward completes
}

// Multiple tapes, one per context
#[cfg(feature = "std")]
static GRADIENT_TAPES: LazyLock<Mutex<HashMap<usize, Vec<TapeEntry>>>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    map.insert(0, Vec::new()); // Default context
    Mutex::new(map)
});

#[cfg(not(feature = "std"))]
static GRADIENT_TAPES: LazyLock<Mutex<HashMap<usize, Vec<TapeEntry>>>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    map.insert(0, Vec::new()); // Default context
    Mutex::new(map)
});

static CONTEXT_COUNTER: AtomicUsize = AtomicUsize::new(1); // Start from 1 (0 is default)

#[cfg(feature = "std")]
thread_local! {
    static CONTEXT_STACK: RefCell<Vec<usize>> = const { RefCell::new(vec![]) };
}

#[cfg(not(feature = "std"))]
static CONTEXT_STACK: Mutex<Vec<usize>> = Mutex::new(Vec::new());

// Helper functions for context stack access
#[cfg(feature = "std")]
fn push_context(context_id: usize) {
    CONTEXT_STACK.with(|stack| {
        stack.borrow_mut().push(context_id);
    });
}

#[cfg(not(feature = "std"))]
fn push_context(context_id: usize) {
    CONTEXT_STACK.lock().push(context_id);
}

#[cfg(feature = "std")]
fn pop_context() {
    CONTEXT_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });
}

#[cfg(not(feature = "std"))]
fn pop_context() {
    CONTEXT_STACK.lock().pop();
}

static IS_COMPUTING_GRADIENTS: AtomicBool = AtomicBool::new(false);

/// Auto-cleaning gradient context using RAII
pub struct GradientContext {
    context_id: usize,
}

impl Default for GradientContext {
    fn default() -> Self {
        let context_id = CONTEXT_COUNTER.fetch_add(1, Ordering::Relaxed);

        // Push to context stack
        push_context(context_id);

        // Initialize empty tape
        #[cfg(feature = "std")]
        {
            GRADIENT_TAPES.lock().unwrap().insert(context_id, Vec::new());
        }
        #[cfg(not(feature = "std"))]
        {
            GRADIENT_TAPES.lock().insert(context_id, Vec::new());
        }

        Self { context_id }
    }
}

impl GradientContext {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Drop for GradientContext {
    fn drop(&mut self) {
        // Don't drop the default context (ID 0)
        if self.context_id == 0 {
            return;
        }

        // Pop from context stack
        pop_context();

        // Remove tape
        #[cfg(feature = "std")]
        {
            GRADIENT_TAPES.lock().unwrap().remove(&self.context_id);
        }
        #[cfg(not(feature = "std"))]
        {
            GRADIENT_TAPES.lock().remove(&self.context_id);
        }
    }
}

#[cfg(feature = "std")]
fn get_active_context() -> usize {
    CONTEXT_STACK.with(|stack| {
        stack.borrow().last().copied().unwrap_or(0) // Default to context 0
    })
}

#[cfg(not(feature = "std"))]
fn get_active_context() -> usize {
    CONTEXT_STACK.lock().last().copied().unwrap_or(0)
}

pub fn is_computing_gradients() -> bool {
    IS_COMPUTING_GRADIENTS.load(Ordering::Relaxed)
}

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
    if !is_computing_gradients() {
        let context_id = get_active_context();

        // Create Tensor handles from input_ids and output_id to keep them alive
        let input_tensors: Vec<Tensor> = input_ids.iter().map(|&id| crate::tensor::tensor_from_id(id)).collect();
        let output_tensor = crate::tensor::tensor_from_id(output_id);

        let mut tapes = {
            #[cfg(feature = "std")]
            {
                GRADIENT_TAPES.lock().map_err(|_| HoduError::GradientTapeCorrupted)?
            }
            #[cfg(not(feature = "std"))]
            {
                GRADIENT_TAPES.lock()
            }
        };

        if let Some(tape) = tapes.get_mut(&context_id) {
            tape.push(TapeEntry {
                output_id,
                op,
                op_params,
                input_ids,
                input_tensors,
                output_tensor,
            });
        }
    }
    Ok(())
}

/// Clear the default context tape (context 0)
pub fn clear_default_context_tape() {
    #[cfg(feature = "std")]
    {
        if let Ok(mut tapes) = GRADIENT_TAPES.lock() {
            if let Some(tape) = tapes.get_mut(&0) {
                tape.clear();
            }
        }
    }
    #[cfg(not(feature = "std"))]
    {
        let mut tapes = GRADIENT_TAPES.lock();
        if let Some(tape) = tapes.get_mut(&0) {
            tape.clear();
        }
    }
}

/// Clear tape for active context
pub fn clear_tape() {
    let context_id = get_active_context();

    #[cfg(feature = "std")]
    {
        if let Ok(mut tapes) = GRADIENT_TAPES.lock() {
            if let Some(tape) = tapes.get_mut(&context_id) {
                tape.clear();
            }
        }
    }
    #[cfg(not(feature = "std"))]
    {
        let mut tapes = GRADIENT_TAPES.lock();
        if let Some(tape) = tapes.get_mut(&context_id) {
            tape.clear();
        }
    }
}

// RAII guard to manage gradient computation lifecycle
struct GradientComputationGuard {
    keep_alive: Vec<Tensor>,
}

impl GradientComputationGuard {
    fn new() -> Self {
        IS_COMPUTING_GRADIENTS.store(true, Ordering::Relaxed);
        Self { keep_alive: Vec::new() }
    }

    fn keep(&mut self, tensor: Tensor) {
        self.keep_alive.push(tensor);
    }
}

impl Drop for GradientComputationGuard {
    fn drop(&mut self) {
        // Set flag to false BEFORE dropping tensors
        IS_COMPUTING_GRADIENTS.store(false, Ordering::Relaxed);
        // Now self.keep_alive will be dropped, and tensors can be properly cleaned up
    }
}

pub fn compute_gradients(loss_tensor_id: TensorId) -> HoduResult<()> {
    let mut guard = GradientComputationGuard::new();

    let result = (|| -> HoduResult<()> {
        let loss_tensor = tensor_from_id(loss_tensor_id);
        let loss_shape = loss_tensor.shape();

        let loss_grad = Tensor::ones(&loss_shape, loss_tensor.dtype())?;
        let loss_grad_id = loss_grad.id();

        set_grad_tensor_id(loss_tensor_id, loss_grad_id)?;

        let mut gradients: HashMap<TensorId, TensorId> = HashMap::new();
        gradients.insert(loss_tensor_id, loss_grad_id);

        guard.keep(loss_tensor);
        guard.keep(loss_grad);

        // Get tape from active context
        let context_id = get_active_context();
        let tape = {
            #[cfg(feature = "std")]
            {
                let tapes = GRADIENT_TAPES.lock().map_err(|_| HoduError::GradientTapeCorrupted)?;
                tapes.get(&context_id).cloned().unwrap_or_default()
            }
            #[cfg(not(feature = "std"))]
            {
                let tapes = GRADIENT_TAPES.lock();
                tapes.get(&context_id).cloned().unwrap_or_default()
            }
        };

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
                        let result_id = create_add_tensor(existing_grad, *grad_id)?;
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
                }
            }
        }

        Ok(())
    })();

    // Explicitly drop guard to set IS_COMPUTING_GRADIENTS = false
    // and clean up keep_alive Vec before returning
    drop(guard);

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
