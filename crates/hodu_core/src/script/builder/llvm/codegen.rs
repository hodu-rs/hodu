use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    ops::Op,
    script::snapshot::{Snapshot, SnapshotNode, SnapshotTensorId},
    types::{DType, Device, Runtime},
};
use inkwell::{
    builder::Builder as LLVMBuilder,
    context::Context,
    module::Module,
    types::{BasicMetadataTypeEnum, BasicTypeEnum},
    values::{BasicMetadataValueEnum, BasicValueEnum, FunctionValue, PointerValue},
    AddressSpace,
};

/// LLVM code generator for Snapshot
pub struct CodeGenerator<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: LLVMBuilder<'ctx>,
    /// Maps SnapshotTensorId to LLVM pointer values (tensor buffers)
    tensor_values: HashMap<SnapshotTensorId, PointerValue<'ctx>>,
    /// Target device for kernel selection
    device: Device,
    /// Target runtime for backend selection
    runtime: Runtime,
    /// Target pointer width in bits (32 or 64)
    pointer_width: u32,
}

impl<'ctx> CodeGenerator<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str, device: Device, runtime: Runtime) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();

        // Determine pointer width from target triple
        let pointer_width = Self::get_pointer_width_from_target(&module);

        Self {
            context,
            module,
            builder,
            tensor_values: HashMap::new(),
            device,
            runtime,
            pointer_width,
        }
    }

    /// Extract pointer width from target triple
    fn get_pointer_width_from_target(module: &Module) -> u32 {
        let triple = module.get_triple();
        let triple_str = triple.as_str().to_string_lossy();

        // Parse target triple (e.g., "x86_64-unknown-linux-gnu", "armv7-unknown-linux")
        // Architecture prefix determines pointer width
        if triple_str.starts_with("x86_64")
            || triple_str.starts_with("aarch64")
            || triple_str.starts_with("arm64")
            || triple_str.starts_with("powerpc64")
            || triple_str.starts_with("riscv64")
            || triple_str.starts_with("s390x")
            || triple_str.starts_with("mips64")
        {
            64
        } else if triple_str.starts_with("i386")
            || triple_str.starts_with("i586")
            || triple_str.starts_with("i686")
            || triple_str.starts_with("x86")
            || triple_str.starts_with("arm")
            || triple_str.starts_with("armv7")
            || triple_str.starts_with("thumbv7")
            || triple_str.starts_with("mips")
            || triple_str.starts_with("powerpc")
            || triple_str.starts_with("riscv32")
        {
            32
        } else {
            // Default to 64-bit for unknown architectures
            64
        }
    }

    /// Get integer type matching target pointer width
    fn get_int_type(&self) -> inkwell::types::IntType<'ctx> {
        match self.pointer_width {
            32 => self.context.i32_type(),
            64 => self.context.i64_type(),
            _ => self.context.i64_type(), // Fallback to i64
        }
    }

    /// Generate LLVM IR from Snapshot
    pub fn generate(&mut self, snapshot: &Snapshot) -> HoduResult<()> {
        // 1. Create function signature from inputs/outputs
        let func = self.create_function(snapshot)?;

        // 2. Create entry basic block
        let entry_block = self.context.append_basic_block(func, "entry");
        self.builder.position_at_end(entry_block);

        // 3. Map input parameters to SnapshotTensorIds
        self.map_function_inputs(snapshot, func)?;

        // 4. Generate instructions for each node
        for node in &snapshot.nodes {
            self.generate_node(node)?;
        }

        // 5. Generate return instruction
        self.generate_return()?;

        Ok(())
    }

    /// Create LLVM function signature from Snapshot
    fn create_function(&mut self, snapshot: &Snapshot) -> HoduResult<FunctionValue<'ctx>> {
        // Function signature: void compute(void* input0, void* input1, ..., void* output0, ...)
        let void_type = self.context.void_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());

        let num_params = snapshot.inputs.len() + snapshot.targets.len();
        let param_types: Vec<BasicMetadataTypeEnum> = vec![ptr_type.into(); num_params];

        let fn_type = void_type.fn_type(&param_types, false);
        let function_name = snapshot.name.as_deref().unwrap_or("compute");
        let function = self.module.add_function(function_name, fn_type, None);

        Ok(function)
    }

    /// Map function input parameters to SnapshotTensorIds
    fn map_function_inputs(&mut self, snapshot: &Snapshot, func: FunctionValue<'ctx>) -> HoduResult<()> {
        // First N params are inputs
        for (i, input) in snapshot.inputs.iter().enumerate() {
            let param = func
                .get_nth_param(i as u32)
                .ok_or_else(|| HoduError::InternalError(format!("Failed to get parameter {}", i)))?;

            if let BasicValueEnum::PointerValue(ptr) = param {
                self.tensor_values.insert(input.id, ptr);
            } else {
                return Err(HoduError::InternalError("Expected pointer parameter".into()));
            }
        }

        // Next M params are outputs (targets)
        let input_count = snapshot.inputs.len();
        for (i, target) in snapshot.targets.iter().enumerate() {
            let param = func
                .get_nth_param((input_count + i) as u32)
                .ok_or_else(|| HoduError::InternalError(format!("Failed to get parameter {}", input_count + i)))?;

            if let BasicValueEnum::PointerValue(ptr) = param {
                self.tensor_values.insert(target.id, ptr);
            } else {
                return Err(HoduError::InternalError("Expected pointer parameter".into()));
            }
        }

        Ok(())
    }

    /// Generate LLVM IR for a single SnapshotNode
    fn generate_node(&mut self, node: &SnapshotNode) -> HoduResult<()> {
        // 1. Get kernel name based on op, device, runtime
        let kernel_name = self.get_kernel_name(&node.op)?;

        // 2. Get or declare the external kernel function
        let kernel_fn = self.get_or_declare_kernel(&kernel_name, node)?;

        // 3. Prepare arguments: input pointers
        let mut args: Vec<BasicMetadataValueEnum> = Vec::new();
        for input_id in &node.input_ids {
            let ptr = self
                .tensor_values
                .get(input_id)
                .ok_or_else(|| HoduError::InternalError(format!("Tensor {:?} not found", input_id)))?;
            args.push(BasicMetadataValueEnum::PointerValue(*ptr));
        }

        // 4. Allocate or get output buffer
        let output_ptr = self.get_or_allocate_output(node)?;
        args.push(BasicMetadataValueEnum::PointerValue(output_ptr));

        // 5. Add metadata (shape, strides, etc.)
        let metadata_ptr = self.create_metadata_constant(node)?;
        args.push(BasicMetadataValueEnum::PointerValue(metadata_ptr));

        // 6. Call the kernel
        self.builder
            .build_call(kernel_fn, &args, "kernel_call")
            .map_err(|e| HoduError::CompilationError(format!("Failed to build call: {}", e)))?;

        Ok(())
    }

    /// Get kernel function name based on op, device, and runtime
    fn get_kernel_name(&self, op: &Op) -> HoduResult<String> {
        // Format: "<runtime>_<device>_<op>"
        // Example: "hodu_cpu_add", "hodu_cuda_matmul"

        let runtime_prefix = match self.runtime {
            Runtime::HODU => "hodu",
            #[cfg(feature = "xla")]
            Runtime::XLA => "xla",
        };

        let device_prefix = match self.device {
            Device::CPU => "cpu",
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => "cuda",
            #[cfg(feature = "metal")]
            Device::METAL => "metal",
        };

        // Convert Op to snake_case name
        let op_name = format!("{:?}", op).to_lowercase();

        Ok(format!("{}_{}_{}", runtime_prefix, device_prefix, op_name))
    }

    /// Get or declare external kernel function
    fn get_or_declare_kernel(&mut self, name: &str, node: &SnapshotNode) -> HoduResult<FunctionValue<'ctx>> {
        // Check if already declared
        if let Some(func) = self.module.get_function(name) {
            return Ok(func);
        }

        // Declare external function
        // Signature: void kernel(void* in0, void* in1, ..., void* out, void* metadata)
        let void_type = self.context.void_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());

        let num_inputs = node.input_ids.len();
        let param_types: Vec<BasicMetadataTypeEnum> = vec![ptr_type.into(); num_inputs + 2]; // inputs + output + metadata

        let fn_type = void_type.fn_type(&param_types, false);
        let function = self.module.add_function(name, fn_type, None);

        Ok(function)
    }

    /// Get or allocate output buffer for a node
    fn get_or_allocate_output(&mut self, node: &SnapshotNode) -> HoduResult<PointerValue<'ctx>> {
        // Check if already allocated (e.g., if it's a target output)
        if let Some(ptr) = self.tensor_values.get(&node.output_id) {
            return Ok(*ptr);
        }

        // Allocate new buffer using alloca
        let buffer_size = node.output_layout.buffer_size();
        let i8_type = self.context.i8_type();

        let array_type = i8_type.array_type(buffer_size as u32);
        let alloca = self
            .builder
            .build_alloca(array_type, "temp_buffer")
            .map_err(|e| HoduError::CompilationError(format!("Failed to allocate buffer: {}", e)))?;

        // Store for future reference
        self.tensor_values.insert(node.output_id, alloca);

        Ok(alloca)
    }

    /// Convert DType to LLVM type
    #[allow(dead_code)]
    fn dtype_to_llvm_type(&self, dtype: DType) -> BasicTypeEnum<'ctx> {
        match dtype {
            DType::BOOL => self.context.bool_type().into(),
            DType::F8E4M3 => self.context.i8_type().into(),
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => self.context.i8_type().into(),
            DType::BF16 => self.context.i16_type().into(),
            DType::F16 => self.context.f16_type().into(),
            DType::F32 => self.context.f32_type().into(),
            #[cfg(feature = "f64")]
            DType::F64 => self.context.f64_type().into(),
            DType::U8 => self.context.i8_type().into(),
            #[cfg(feature = "u16")]
            DType::U16 => self.context.i16_type().into(),
            DType::U32 => self.context.i32_type().into(),
            #[cfg(feature = "u64")]
            DType::U64 => self.context.i64_type().into(),
            DType::I8 => self.context.i8_type().into(),
            #[cfg(feature = "i16")]
            DType::I16 => self.context.i16_type().into(),
            DType::I32 => self.context.i32_type().into(),
            #[cfg(feature = "i64")]
            DType::I64 => self.context.i64_type().into(),
        }
    }

    /// Create metadata constant (shape, strides, offset info)
    fn create_metadata_constant(&mut self, node: &SnapshotNode) -> HoduResult<PointerValue<'ctx>> {
        // Build metadata array: [num_inputs, ndim, shape..., strides..., offset, ...]
        let mut metadata = Vec::new();

        // Number of inputs
        metadata.push(node.input_ids.len() as u64);

        // For each input: ndim, shape, strides, offset
        for layout in &node.input_layouts {
            metadata.push(layout.ndim() as u64);
            for &dim in layout.shape().dims() {
                metadata.push(dim as u64);
            }
            for &stride in layout.strides() {
                metadata.push(stride as u64);
            }
            metadata.push(layout.offset() as u64);
        }

        // Output: ndim, shape, strides, offset
        metadata.push(node.output_layout.ndim() as u64);
        for &dim in node.output_layout.shape().dims() {
            metadata.push(dim as u64);
        }
        for &stride in node.output_layout.strides() {
            metadata.push(stride as u64);
        }
        metadata.push(node.output_layout.offset() as u64);

        // Create constant array using target-specific integer type
        let int_type = self.get_int_type();
        let array_type = int_type.array_type(metadata.len() as u32);

        // Create the constant array from the values
        let const_array = int_type.const_array(
            &metadata
                .iter()
                .map(|&v| int_type.const_int(v, false))
                .collect::<Vec<_>>(),
        );

        // Create global constant
        let global = self.module.add_global(array_type, None, "metadata");
        global.set_initializer(&const_array);
        global.set_constant(true);

        Ok(global.as_pointer_value())
    }

    /// Generate return instruction
    fn generate_return(&mut self) -> HoduResult<()> {
        self.builder
            .build_return(None)
            .map_err(|e| HoduError::CompilationError(format!("Failed to build return: {}", e)))?;
        Ok(())
    }

    /// Get the generated LLVM module
    pub fn module(&self) -> &Module<'ctx> {
        &self.module
    }
}
