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
    AddressSpace, OptimizationLevel,
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
            64 => {
                // Check if 64-bit features are enabled
                #[cfg(any(feature = "f64", feature = "u64", feature = "i64"))]
                {
                    self.context.i64_type()
                }
                #[cfg(not(any(feature = "f64", feature = "u64", feature = "i64")))]
                {
                    // If no 64-bit features enabled, fall back to i32
                    self.context.i32_type()
                }
            },
            _ => self.context.i32_type(), // Fallback to i32 for unknown widths
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

        // 4. Load constant tensors as global variables
        self.load_constants(snapshot)?;

        // 5. Generate instructions for each node
        for node in &snapshot.nodes {
            self.generate_node(node)?;
        }

        // 6. Generate return instruction
        self.generate_return()?;

        Ok(())
    }

    /// Verify the generated LLVM module
    pub fn verify(&self) -> HoduResult<()> {
        if let Err(err) = self.module.verify() {
            return Err(HoduError::CompilationError(format!(
                "LLVM module verification failed: {}",
                err
            )));
        }
        Ok(())
    }

    /// Apply optimization passes to the module
    /// Note: Full optimization support requires newer LLVM/inkwell versions
    /// This is a placeholder for future optimization implementation
    pub fn optimize(&self, _opt_level: OptimizationLevel) -> HoduResult<()> {
        // TODO: Implement optimization passes when using newer inkwell version
        // For now, optimization will be handled at the TargetMachine level during code generation
        Ok(())
    }

    /// Verify and optimize the module
    pub fn verify_and_optimize(&self, opt_level: OptimizationLevel) -> HoduResult<()> {
        // First verify the module
        self.verify()?;

        // Then optimize
        self.optimize(opt_level)?;

        Ok(())
    }

    /// Print LLVM IR to string (for debugging)
    pub fn print_to_string(&self) -> String {
        self.module.print_to_string().to_string()
    }

    /// Print LLVM IR to file
    pub fn print_to_file(&self, path: &str) -> HoduResult<()> {
        self.module
            .print_to_file(path)
            .map_err(|e| HoduError::CompilationError(format!("Failed to write LLVM IR to file: {}", e)))?;
        Ok(())
    }

    /// Emit object file (.o) from LLVM IR
    #[cfg(feature = "std")]
    pub fn emit_object_file(&self, path: &str, opt_level: OptimizationLevel) -> HoduResult<()> {
        use inkwell::targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine};
        use std::path::Path;

        // 1. Initialize native target
        Target::initialize_native(&InitializationConfig::default())
            .map_err(|e| HoduError::CompilationError(format!("Failed to initialize native target: {}", e)))?;

        // 2. Get target triple
        let target_triple = TargetMachine::get_default_triple();

        // 3. Get target from triple
        let target = Target::from_triple(&target_triple)
            .map_err(|e| HoduError::CompilationError(format!("Failed to get target from triple: {}", e)))?;

        // 4. Get CPU and features
        let cpu = TargetMachine::get_host_cpu_name().to_string();
        let features = TargetMachine::get_host_cpu_features().to_string();

        // 5. Create target machine
        let target_machine = target
            .create_target_machine(
                &target_triple,
                &cpu,
                &features,
                opt_level,
                RelocMode::PIC, // Position Independent Code for shared libraries
                CodeModel::Default,
            )
            .ok_or_else(|| HoduError::CompilationError("Failed to create target machine".into()))?;

        // 6. Write object file
        target_machine
            .write_to_file(&self.module, FileType::Object, Path::new(path))
            .map_err(|e| HoduError::CompilationError(format!("Failed to write object file: {}", e)))?;

        Ok(())
    }

    /// Emit shared library (.so/.dylib/.dll) from LLVM IR
    #[cfg(feature = "std")]
    pub fn emit_shared_library(&self, path: &str, opt_level: OptimizationLevel) -> HoduResult<()> {
        use std::process::Command;

        // 1. Create temporary object file
        let temp_obj = format!("{}.o", path);
        self.emit_object_file(&temp_obj, opt_level)?;

        // 2. Platform-specific linker command and invocation
        let status = if cfg!(target_os = "linux") {
            Command::new("gcc")
                .args(&["-shared", "-fPIC", "-o", path, &temp_obj])
                .status()
                .map_err(|e| HoduError::CompilationError(format!("Failed to invoke gcc: {}", e)))?
        } else if cfg!(target_os = "macos") {
            Command::new("clang")
                .args(&["-shared", "-o", path, &temp_obj])
                .status()
                .map_err(|e| HoduError::CompilationError(format!("Failed to invoke clang: {}", e)))?
        } else if cfg!(target_os = "windows") {
            let dll_out = format!("/OUT:{}", path);
            Command::new("link")
                .args(&["/DLL", &dll_out, &temp_obj])
                .status()
                .map_err(|e| HoduError::CompilationError(format!("Failed to invoke link: {}", e)))?
        } else {
            return Err(HoduError::UnsupportedPlatform(
                "Object file emission not supported on this platform".into(),
            ));
        };

        // 3. Check linker status
        if !status.success() {
            return Err(HoduError::CompilationError(format!(
                "Linker failed with exit code: {:?}",
                status.code()
            )));
        }

        // 4. Clean up temporary object file
        std::fs::remove_file(&temp_obj)
            .map_err(|e| HoduError::IoError(format!("Failed to remove temporary object file: {}", e)))?;

        Ok(())
    }

    /// Emit assembly file (.s) from LLVM IR (for debugging)
    #[cfg(feature = "std")]
    pub fn emit_assembly(&self, path: &str, opt_level: OptimizationLevel) -> HoduResult<()> {
        use inkwell::targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine};
        use std::path::Path;

        // Initialize and create target machine (same as emit_object_file)
        Target::initialize_native(&InitializationConfig::default())
            .map_err(|e| HoduError::CompilationError(format!("Failed to initialize native target: {}", e)))?;

        let target_triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&target_triple)
            .map_err(|e| HoduError::CompilationError(format!("Failed to get target: {}", e)))?;

        let cpu = TargetMachine::get_host_cpu_name().to_string();
        let features = TargetMachine::get_host_cpu_features().to_string();

        let target_machine = target
            .create_target_machine(
                &target_triple,
                &cpu,
                &features,
                opt_level,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .ok_or_else(|| HoduError::CompilationError("Failed to create target machine".into()))?;

        // Write assembly file
        target_machine
            .write_to_file(&self.module, FileType::Assembly, Path::new(path))
            .map_err(|e| HoduError::CompilationError(format!("Failed to write assembly file: {}", e)))?;

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

    /// Load constant tensors as LLVM global variables
    fn load_constants(&mut self, snapshot: &Snapshot) -> HoduResult<()> {
        for constant in &snapshot.constants {
            // Create global variable name
            let global_name = format!("const_tensor_{}", constant.id.0);

            // Determine element type and count
            let element_type = self.dtype_to_llvm_type(constant.dtype);
            let num_elements = constant.shape.size();

            // Create array type for the constant data
            let array_type = match element_type {
                BasicTypeEnum::IntType(int_type) => int_type.array_type(num_elements as u32),
                BasicTypeEnum::FloatType(float_type) => float_type.array_type(num_elements as u32),
                _ => {
                    return Err(HoduError::InternalError(format!(
                        "Unsupported constant dtype: {:?}",
                        constant.dtype
                    )))
                },
            };

            // Create global variable
            let global = self.module.add_global(array_type, None, &global_name);
            global.set_constant(true);
            global.set_unnamed_addr(true);

            // Initialize with constant data
            self.initialize_constant_data(global, constant)?;

            // Store pointer in tensor_values map
            let ptr = global.as_pointer_value();
            self.tensor_values.insert(constant.id, ptr);
        }

        Ok(())
    }

    /// Initialize global constant with data from SnapshotConstant
    fn initialize_constant_data(
        &self,
        global: inkwell::values::GlobalValue<'ctx>,
        constant: &crate::script::SnapshotConstant,
    ) -> HoduResult<()> {
        // Helper function to convert bytes to typed values
        fn bytes_to_values<T: Copy>(data: &[u8]) -> Vec<T> {
            let num_elements = data.len() / core::mem::size_of::<T>();
            let mut values = Vec::with_capacity(num_elements);
            unsafe {
                let ptr = data.as_ptr() as *const T;
                for i in 0..num_elements {
                    values.push(*ptr.add(i));
                }
            }
            values
        }

        // Create constant array based on dtype (in dtype.rs definition order)
        let const_array = match constant.dtype {
            DType::BOOL => {
                let values = bytes_to_values::<bool>(&constant.data);
                let const_values: Vec<_> = values
                    .iter()
                    .map(|&v| self.context.bool_type().const_int(v as u64, false))
                    .collect();
                self.context.bool_type().const_array(&const_values)
            },
            DType::F8E4M3 => {
                // F8E4M3 stored as i8
                let values = bytes_to_values::<u8>(&constant.data);
                let const_values: Vec<_> = values
                    .iter()
                    .map(|&v| self.context.i8_type().const_int(v as u64, false))
                    .collect();
                self.context.i8_type().const_array(&const_values)
            },
            #[cfg(feature = "f8e5m2")]
            DType::F8E5M2 => {
                // F8E5M2 stored as i8
                let values = bytes_to_values::<u8>(&constant.data);
                let const_values: Vec<_> = values
                    .iter()
                    .map(|&v| self.context.i8_type().const_int(v as u64, false))
                    .collect();
                self.context.i8_type().const_array(&const_values)
            },
            DType::BF16 => {
                // BF16 stored as i16
                let values = bytes_to_values::<u16>(&constant.data);
                let const_values: Vec<_> = values
                    .iter()
                    .map(|&v| self.context.i16_type().const_int(v as u64, false))
                    .collect();
                self.context.i16_type().const_array(&const_values)
            },
            DType::F16 => {
                let values = bytes_to_values::<half::f16>(&constant.data);
                let const_values: Vec<_> = values
                    .iter()
                    .map(|&v| self.context.f16_type().const_float(v.to_f64()))
                    .collect();
                self.context.f16_type().const_array(&const_values)
            },
            DType::F32 => {
                let values = bytes_to_values::<f32>(&constant.data);
                let const_values: Vec<_> = values
                    .iter()
                    .map(|&v| self.context.f32_type().const_float(v as f64))
                    .collect();
                self.context.f32_type().const_array(&const_values)
            },
            #[cfg(feature = "f64")]
            DType::F64 => {
                let values = bytes_to_values::<f64>(&constant.data);
                let const_values: Vec<_> = values.iter().map(|&v| self.context.f64_type().const_float(v)).collect();
                self.context.f64_type().const_array(&const_values)
            },
            DType::U8 => {
                let const_values: Vec<_> = constant
                    .data
                    .iter()
                    .map(|&v| self.context.i8_type().const_int(v as u64, false))
                    .collect();
                self.context.i8_type().const_array(&const_values)
            },
            #[cfg(feature = "u16")]
            DType::U16 => {
                let values = bytes_to_values::<u16>(&constant.data);
                let const_values: Vec<_> = values
                    .iter()
                    .map(|&v| self.context.i16_type().const_int(v as u64, false))
                    .collect();
                self.context.i16_type().const_array(&const_values)
            },
            DType::U32 => {
                let values = bytes_to_values::<u32>(&constant.data);
                let const_values: Vec<_> = values
                    .iter()
                    .map(|&v| self.context.i32_type().const_int(v as u64, false))
                    .collect();
                self.context.i32_type().const_array(&const_values)
            },
            #[cfg(feature = "u64")]
            DType::U64 => {
                let values = bytes_to_values::<u64>(&constant.data);
                let const_values: Vec<_> = values
                    .iter()
                    .map(|&v| self.context.i64_type().const_int(v, false))
                    .collect();
                self.context.i64_type().const_array(&const_values)
            },
            DType::I8 => {
                let values = bytes_to_values::<i8>(&constant.data);
                let const_values: Vec<_> = values
                    .iter()
                    .map(|&v| self.context.i8_type().const_int(v as u64, true))
                    .collect();
                self.context.i8_type().const_array(&const_values)
            },
            #[cfg(feature = "i16")]
            DType::I16 => {
                let values = bytes_to_values::<i16>(&constant.data);
                let const_values: Vec<_> = values
                    .iter()
                    .map(|&v| self.context.i16_type().const_int(v as u64, true))
                    .collect();
                self.context.i16_type().const_array(&const_values)
            },
            DType::I32 => {
                let values = bytes_to_values::<i32>(&constant.data);
                let const_values: Vec<_> = values
                    .iter()
                    .map(|&v| self.context.i32_type().const_int(v as u64, true))
                    .collect();
                self.context.i32_type().const_array(&const_values)
            },
            #[cfg(feature = "i64")]
            DType::I64 => {
                let values = bytes_to_values::<i64>(&constant.data);
                let const_values: Vec<_> = values
                    .iter()
                    .map(|&v| self.context.i64_type().const_int(v as u64, true))
                    .collect();
                self.context.i64_type().const_array(&const_values)
            },
        };

        global.set_initializer(&const_array);
        Ok(())
    }

    /// Generate LLVM IR for a single SnapshotNode
    fn generate_node(&mut self, node: &SnapshotNode) -> HoduResult<()> {
        // 1. Get kernel name based on op and dtype
        let kernel_name = self.get_kernel_name(&node.op, node.output_dtype)?;

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

    /// Get kernel function name based on runtime, device, op and dtype
    fn get_kernel_name(&self, op: &Op, dtype: DType) -> HoduResult<String> {
        // Format: "{runtime}_{device}_{op}_{dtype}"
        // Example: "hodu_cpu_add_f32", "xla_cuda_matmul_f16"

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

        Ok(format!("{}_{}_{}_{}", runtime_prefix, device_prefix, op, dtype))
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
