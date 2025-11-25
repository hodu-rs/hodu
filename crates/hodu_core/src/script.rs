pub mod builder;
pub mod capture;
pub mod compiled;
pub mod metadata;
pub mod snapshot;

use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    tensor::Tensor,
    types::{Device, HoduRuntimeCompiler, Runtime},
};
pub use builder::{BuildConfig, BuildType, Builder, TargetArch, TargetConfig, TargetEnv, TargetOS, TargetVendor};
pub use capture::{CaptureBoard, CaptureBoardId, CapturedInput, CapturedOp, CapturedTarget};
pub use compiled::{CompiledState, HoduCompiledState, LLVMJitState};
pub use snapshot::{Snapshot, SnapshotConstant, SnapshotInput, SnapshotNode, SnapshotTarget, SnapshotTensorId};

/// Script holds the Hodu Script IR and provides compilation/execution interface
pub struct Script {
    snapshot: Snapshot,
    compiled: Option<CompiledState>,
    device: Option<Device>,
    runtime: Option<Runtime>,
    compiler: Option<HoduRuntimeCompiler>,
}

impl Script {
    /// Create a new Script from a Snapshot
    pub fn new(snapshot: Snapshot) -> Self {
        Self {
            snapshot,
            compiled: None,
            device: None,
            runtime: None,
            compiler: None,
        }
    }

    /// Get reference to the underlying snapshot
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    /// Get mutable reference to the underlying snapshot
    pub fn snapshot_mut(&mut self) -> &mut Snapshot {
        &mut self.snapshot
    }

    /// Consume and return the underlying snapshot
    pub fn into_snapshot(self) -> Snapshot {
        self.snapshot
    }

    /// Set device (clears compiled state)
    pub fn set_device(&mut self, device: Device) {
        self.device = Some(device);
        self.compiled = None;
    }

    /// Set runtime (clears compiled state)
    pub fn set_runtime(&mut self, runtime: Runtime) {
        self.runtime = Some(runtime);
        self.compiled = None;
    }

    /// Set compiler for HODU runtime (clears compiled state)
    pub fn set_compiler(&mut self, compiler: HoduRuntimeCompiler) {
        self.compiler = Some(compiler);
        self.compiled = None;
    }

    /// Get device
    pub fn device(&self) -> Option<Device> {
        self.device
    }

    /// Get runtime
    pub fn runtime(&self) -> Option<Runtime> {
        self.runtime
    }

    /// Get compiler
    pub fn compiler(&self) -> Option<HoduRuntimeCompiler> {
        self.compiler
    }

    /// Save Script to a file
    #[cfg(feature = "serde")]
    pub fn save(&self, path: impl AsRef<str>) -> HoduResult<()> {
        let path_str = path.as_ref();

        if !path_str.ends_with(".hdss") {
            return Err(HoduError::InvalidArgument("File must have .hdss extension".into()));
        }

        let serialized = postcard::to_allocvec(&self.snapshot)
            .map_err(|e| HoduError::SerializationFailed(format!("Postcard serialization failed: {}", e)))?;

        #[cfg(feature = "std")]
        {
            std::fs::write(path_str, serialized)
                .map_err(|e| HoduError::IoError(format!("Failed to write file: {}", e)))?;
        }

        #[cfg(not(feature = "std"))]
        {
            return Err(HoduError::UnsupportedOperation(
                "File I/O not available in no_std environment".into(),
            ));
        }

        Ok(())
    }

    /// Load Script from a file
    #[cfg(feature = "serde")]
    pub fn load(path: impl AsRef<str>) -> HoduResult<Self> {
        let path_str = path.as_ref();

        if !path_str.ends_with(".hdss") {
            return Err(HoduError::InvalidArgument("File must have .hdss extension".into()));
        }

        #[cfg(feature = "std")]
        {
            let bytes =
                std::fs::read(path_str).map_err(|e| HoduError::IoError(format!("Failed to read file: {}", e)))?;

            let snapshot: Snapshot = postcard::from_bytes(&bytes)
                .map_err(|e| HoduError::DeserializationFailed(format!("Postcard deserialization failed: {}", e)))?;

            Ok(Self::new(snapshot))
        }

        #[cfg(not(feature = "std"))]
        {
            Err(HoduError::UnsupportedOperation(
                "File I/O not available in no_std environment".into(),
            ))
        }
    }

    /// Compile the script for execution
    /// This prepares the script for JIT execution by generating LLVM IR or XLA executable
    pub fn compile(&mut self) -> HoduResult<()> {
        let device = self
            .device
            .ok_or_else(|| HoduError::InvalidArgument("Device not set. Use set_device() first.".into()))?;

        let runtime = self
            .runtime
            .ok_or_else(|| HoduError::InvalidArgument("Runtime not set. Use set_runtime() first.".into()))?;

        // Check if runtime supports device
        if !runtime.is_supported(device) {
            return Err(HoduError::UnsupportedOperation(format!(
                "Runtime {:?} does not support device {:?}",
                runtime, device
            )));
        }

        // Compile based on runtime
        let compiled_state = match runtime {
            Runtime::HODU => {
                // Get compiler (default to LLVM if not set)
                let compiler = self.compiler.unwrap_or(HoduRuntimeCompiler::LLVM);

                match compiler {
                    HoduRuntimeCompiler::LLVM => {
                        // Generate LLVM IR and create JIT engine
                        use crate::script::builder::llvm::CodeGenerator;
                        use inkwell::context::Context;
                        use inkwell::OptimizationLevel;

                        // Create context on heap and get a 'static reference
                        let context = Context::create();
                        let context_ptr = Box::into_raw(Box::new(context));

                        // SAFETY: We leaked context to get 'static lifetime
                        // It will be reclaimed when LLVMJitState is dropped
                        let context_ref: &'static Context = unsafe { &*context_ptr };

                        let mut codegen = CodeGenerator::new(context_ref, "hodu_jit", device, runtime);

                        // Generate LLVM IR from snapshot
                        codegen.generate(&self.snapshot)?;

                        // DEBUG: Print LLVM IR
                        #[cfg(feature = "std")]
                        {
                            if let Ok(ir_string) = codegen.module().print_to_string().to_str() {
                                println!("[DEBUG] Generated LLVM IR:\n{}", ir_string);
                            }
                        }

                        // Create JIT execution engine with kernel symbols registered
                        let engine = codegen.create_jit_engine(OptimizationLevel::Default, &self.snapshot)?;

                        // SAFETY: We manually manage context and engine lifetimes
                        // Context is already leaked, so we reconstruct it from the pointer
                        let context_owned = unsafe { *Box::from_raw(context_ptr) };
                        let llvm_state = unsafe { LLVMJitState::new(context_owned, engine, &self.snapshot) };

                        CompiledState::HODU(HoduCompiledState::LLVM(llvm_state))
                    },
                }
            },

            #[cfg(feature = "xla")]
            Runtime::XLA => {
                // TODO: Compile to XLA executable
                return Err(HoduError::UnsupportedOperation(
                    "XLA compilation not yet implemented".into(),
                ));
            },
        };

        self.compiled = Some(compiled_state);
        Ok(())
    }

    /// Run the script with inputs and return outputs as HashMap<target_name, Tensor>
    /// Automatically compiles if not already compiled
    pub fn run(&mut self, inputs: &[(&str, &Tensor)]) -> HoduResult<HashMap<String, Tensor>> {
        // Auto-compile if needed
        if self.compiled.is_none() {
            self.compile()?;
        }

        // TODO: Validate inputs match snapshot.inputs

        // Execute using the compiled state
        let compiled_state = self
            .compiled
            .as_ref()
            .ok_or_else(|| HoduError::InternalError("Compiled state should exist after compile()".into()))?;

        compiled_state.execute(inputs)
    }
}
