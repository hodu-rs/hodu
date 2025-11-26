use crate::compat::*;
use crate::error::{HoduError, HoduResult};
use crate::script::snapshot::{Snapshot, SnapshotInput};
use crate::tensor::Tensor;
use crate::types::{DType, Layout};
use core::mem::ManuallyDrop;
use inkwell::{context::Context, execution_engine::ExecutionEngine};

/// Output specification with layout and dtype
#[derive(Debug, Clone)]
struct OutputSpec {
    name: String,
    layout: Layout,
    dtype: DType,
}

/// LLVM-compiled state holding context and JIT execution engine
/// Uses 'static lifetime via Box::leak pattern for self-referential structure
pub struct LLVMJitState {
    /// Leaked context pointer - will be reclaimed on drop
    context_ptr: *mut Context,
    /// JIT execution engine with 'static lifetime (actually references context_ptr)
    engine: ManuallyDrop<ExecutionEngine<'static>>,
    /// Function name in the module
    function_name: String,
    /// Input specifications from snapshot
    inputs: Vec<SnapshotInput>,
    /// Output specifications (computed from snapshot)
    outputs: Vec<OutputSpec>,
}

impl LLVMJitState {
    /// Create new LLVM JIT state
    /// SAFETY: Caller must ensure context and engine lifetimes are properly managed
    pub unsafe fn new(context: Context, engine: ExecutionEngine<'_>, snapshot: &Snapshot) -> Self {
        // Leak context to get 'static lifetime
        let context_ptr = Box::into_raw(Box::new(context));

        // Transmute engine to 'static (it actually references context_ptr)
        let engine: ExecutionEngine<'static> = core::mem::transmute(engine);

        // Compute output specs from snapshot
        let outputs = Self::compute_output_specs(snapshot);

        Self {
            context_ptr,
            engine: ManuallyDrop::new(engine),
            function_name: snapshot.name.clone().unwrap_or_else(|| "compute".to_string()),
            inputs: snapshot.inputs.clone(),
            outputs,
        }
    }

    /// Compute output specifications from snapshot targets
    fn compute_output_specs(snapshot: &Snapshot) -> Vec<OutputSpec> {
        snapshot
            .targets
            .iter()
            .filter_map(|target| {
                // Find the node that produces this target
                snapshot
                    .nodes
                    .iter()
                    .find(|node| node.output_id == target.id)
                    .map(|node| OutputSpec {
                        name: target.name.clone(),
                        layout: node.output_layout.clone(),
                        dtype: node.output_dtype,
                    })
            })
            .collect()
    }

    /// Execute the compiled function with inputs and return outputs
    pub fn execute(&self, inputs: &[(&str, &Tensor)]) -> HoduResult<HashMap<String, Tensor>> {
        // 1. Validate and order inputs according to snapshot
        let ordered_inputs = self.validate_and_order_inputs(inputs)?;

        // 2. Get function address from execution engine
        let fn_addr = self
            .engine
            .get_function_address(&self.function_name)
            .map_err(|e| HoduError::CompilationError(format!("Failed to get function address: {}", e)))?;

        // 3. Prepare input pointers
        let input_ptrs: Vec<*const u8> = ordered_inputs.iter().map(|tensor| tensor.data_ptr()).collect();

        // 4. Allocate output buffers
        let output_tensors = self.allocate_output_tensors()?;
        let output_ptrs: Vec<*mut u8> = output_tensors
            .iter()
            .map(|tensor| tensor.data_ptr() as *mut u8)
            .collect();

        // 5. Call JIT function
        self.call_jit_function(fn_addr, &input_ptrs, &output_ptrs)?;

        // 6. Build output HashMap
        let mut result = HashMap::new();
        for (output_spec, tensor) in self.outputs.iter().zip(output_tensors.into_iter()) {
            result.insert(output_spec.name.clone(), tensor);
        }

        Ok(result)
    }

    /// Validate inputs and order them according to snapshot.inputs
    fn validate_and_order_inputs<'a>(&self, inputs: &'a [(&str, &'a Tensor)]) -> HoduResult<Vec<&'a Tensor>> {
        let mut ordered = Vec::with_capacity(self.inputs.len());

        for snapshot_input in &self.inputs {
            let tensor = inputs
                .iter()
                .find(|(name, _)| *name == snapshot_input.name)
                .map(|(_, t)| *t)
                .ok_or_else(|| HoduError::InvalidArgument(format!("Missing input tensor: {}", snapshot_input.name)))?;

            // Validate shape and dtype
            if tensor.shape() != snapshot_input.shape {
                return Err(HoduError::InvalidArgument(format!(
                    "Input '{}': expected shape {:?}, got {:?}",
                    snapshot_input.name,
                    snapshot_input.shape,
                    tensor.shape()
                )));
            }
            if tensor.dtype() != snapshot_input.dtype {
                return Err(HoduError::InvalidArgument(format!(
                    "Input '{}': expected dtype {:?}, got {:?}",
                    snapshot_input.name,
                    snapshot_input.dtype,
                    tensor.dtype()
                )));
            }

            ordered.push(tensor);
        }

        Ok(ordered)
    }

    /// Allocate output tensors based on target specifications
    fn allocate_output_tensors(&self) -> HoduResult<Vec<Tensor>> {
        self.outputs
            .iter()
            .map(|output_spec| {
                // Allocate uninitialized tensor with the correct layout and dtype
                Tensor::empty(output_spec.layout.shape().clone(), output_spec.dtype)
            })
            .collect()
    }

    /// Call the JIT-compiled function with input and output pointers
    fn call_jit_function(&self, fn_addr: usize, input_ptrs: &[*const u8], output_ptrs: &[*mut u8]) -> HoduResult<()> {
        // Combine input and output pointers
        // Function signature: void compute(void* input0, ..., void* output0, ...)
        let num_inputs = input_ptrs.len();
        let num_outputs = output_ptrs.len();
        let total_params = num_inputs + num_outputs;

        // Helper to get nth pointer (input or output)
        let get_ptr = |idx: usize| -> *const u8 {
            if idx < num_inputs {
                input_ptrs[idx]
            } else {
                output_ptrs[idx - num_inputs] as *const u8
            }
        };

        // Call function based on total parameter count
        // We support up to 12 total parameters
        unsafe {
            match total_params {
                0 => {
                    let func: extern "C" fn() = core::mem::transmute(fn_addr);
                    func();
                },
                1 => {
                    let func: extern "C" fn(*const u8) = core::mem::transmute(fn_addr);
                    func(get_ptr(0));
                },
                2 => {
                    let func: extern "C" fn(*const u8, *const u8) = core::mem::transmute(fn_addr);
                    func(get_ptr(0), get_ptr(1));
                },
                3 => {
                    let func: extern "C" fn(*const u8, *const u8, *const u8) = core::mem::transmute(fn_addr);
                    func(get_ptr(0), get_ptr(1), get_ptr(2));
                },
                4 => {
                    let func: extern "C" fn(*const u8, *const u8, *const u8, *const u8) = core::mem::transmute(fn_addr);
                    func(get_ptr(0), get_ptr(1), get_ptr(2), get_ptr(3));
                },
                5 => {
                    let func: extern "C" fn(*const u8, *const u8, *const u8, *const u8, *const u8) =
                        core::mem::transmute(fn_addr);
                    func(get_ptr(0), get_ptr(1), get_ptr(2), get_ptr(3), get_ptr(4));
                },
                6 => {
                    let func: extern "C" fn(*const u8, *const u8, *const u8, *const u8, *const u8, *const u8) =
                        core::mem::transmute(fn_addr);
                    func(get_ptr(0), get_ptr(1), get_ptr(2), get_ptr(3), get_ptr(4), get_ptr(5));
                },
                7 => {
                    let func: extern "C" fn(
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                    ) = core::mem::transmute(fn_addr);
                    func(
                        get_ptr(0),
                        get_ptr(1),
                        get_ptr(2),
                        get_ptr(3),
                        get_ptr(4),
                        get_ptr(5),
                        get_ptr(6),
                    );
                },
                8 => {
                    let func: extern "C" fn(
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                    ) = core::mem::transmute(fn_addr);
                    func(
                        get_ptr(0),
                        get_ptr(1),
                        get_ptr(2),
                        get_ptr(3),
                        get_ptr(4),
                        get_ptr(5),
                        get_ptr(6),
                        get_ptr(7),
                    );
                },
                9 => {
                    let func: extern "C" fn(
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                    ) = core::mem::transmute(fn_addr);
                    func(
                        get_ptr(0),
                        get_ptr(1),
                        get_ptr(2),
                        get_ptr(3),
                        get_ptr(4),
                        get_ptr(5),
                        get_ptr(6),
                        get_ptr(7),
                        get_ptr(8),
                    );
                },
                10 => {
                    let func: extern "C" fn(
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                    ) = core::mem::transmute(fn_addr);
                    func(
                        get_ptr(0),
                        get_ptr(1),
                        get_ptr(2),
                        get_ptr(3),
                        get_ptr(4),
                        get_ptr(5),
                        get_ptr(6),
                        get_ptr(7),
                        get_ptr(8),
                        get_ptr(9),
                    );
                },
                11 => {
                    let func: extern "C" fn(
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                    ) = core::mem::transmute(fn_addr);
                    func(
                        get_ptr(0),
                        get_ptr(1),
                        get_ptr(2),
                        get_ptr(3),
                        get_ptr(4),
                        get_ptr(5),
                        get_ptr(6),
                        get_ptr(7),
                        get_ptr(8),
                        get_ptr(9),
                        get_ptr(10),
                    );
                },
                12 => {
                    let func: extern "C" fn(
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                        *const u8,
                    ) = core::mem::transmute(fn_addr);
                    func(
                        get_ptr(0),
                        get_ptr(1),
                        get_ptr(2),
                        get_ptr(3),
                        get_ptr(4),
                        get_ptr(5),
                        get_ptr(6),
                        get_ptr(7),
                        get_ptr(8),
                        get_ptr(9),
                        get_ptr(10),
                        get_ptr(11),
                    );
                },
                _ => {
                    return Err(HoduError::UnsupportedOperation(format!(
                        "Unsupported parameter count: {}. Maximum supported: 12",
                        total_params
                    )));
                },
            }
        }

        Ok(())
    }
}

impl Drop for LLVMJitState {
    fn drop(&mut self) {
        // SAFETY: Drop engine first (which references context), then reclaim context
        unsafe {
            ManuallyDrop::drop(&mut self.engine);
            let _ = Box::from_raw(self.context_ptr);
        }
    }
}
