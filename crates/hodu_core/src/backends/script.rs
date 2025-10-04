pub(crate) mod ir;

use crate::error::{HoduError, HoduResult};
use crate::{
    backends::executor::{CompileOptions, CompiledScript, ExecutionOutputs, Executor, ExecutorT},
    compat::*,
    tensor::Tensor,
    types::{backend::Backend, device::Device},
};
use ir::ScriptIR;

pub struct Script {
    ir: Option<ScriptIR>,
    backend: Option<Backend>,
    device: Option<Device>,
    runtime_inputs: HashMap<String, Tensor>,
    compiled: Option<CompiledScript>,
    executor: Option<Executor>,
}

impl Default for Script {
    fn default() -> Self {
        Self {
            ir: None,
            backend: None,
            device: None,
            runtime_inputs: HashMap::new(),
            compiled: None,
            executor: None,
        }
    }
}

impl std::fmt::Debug for Script {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Script")
            .field("ir", &self.ir)
            .field("backend", &self.backend)
            .field("device", &self.device)
            .field("runtime_inputs", &self.runtime_inputs)
            .field("compiled", &self.compiled.is_some())
            .field("executor", &self.executor.is_some())
            .finish()
    }
}

impl Script {
    pub fn new(name: String) -> Self {
        let ir = ScriptIR::new(name);
        Self {
            ir: Some(ir),
            backend: None,
            device: None,
            runtime_inputs: HashMap::new(),
            compiled: None,
            executor: None,
        }
    }

    pub fn with_ir(ir: ScriptIR) -> Self {
        Self {
            ir: Some(ir),
            backend: None,
            device: None,
            runtime_inputs: HashMap::new(),
            compiled: None,
            executor: None,
        }
    }

    pub fn get_name(&self) -> Option<&str> {
        self.ir.as_ref().map(|ir| ir.metadata.name.as_str())
    }

    pub fn set_name(&mut self, name: String) {
        if let Some(ir) = &mut self.ir {
            ir.metadata.name = name;
        }
    }

    pub fn get_ir(&self) -> Option<&ScriptIR> {
        self.ir.as_ref()
    }

    pub fn get_ir_mut(&mut self) -> Option<&mut ScriptIR> {
        self.ir.as_mut()
    }

    pub fn get_backend(&self) -> Option<Backend> {
        self.backend
    }

    pub fn set_backend(&mut self, backend: Backend) {
        self.backend = Some(backend);
        // Invalidate cached compilation when backend changes
        self.compiled = None;
        self.executor = None;
    }

    pub fn get_device(&self) -> Option<Device> {
        self.device
    }

    pub fn set_device(&mut self, device: Device) {
        self.device = Some(device);
        // Invalidate cached compilation when device changes
        self.compiled = None;
        self.executor = None;
    }

    /// Add an input tensor for script execution
    ///
    /// # Arguments
    /// * `name` - Name of the input (must match script's input names)
    /// * `tensor` - Input tensor
    ///
    /// # Example
    /// ```
    /// script.add_input("x", tensor_x);
    /// script.add_input("y", tensor_y);
    /// let result = script.run()?;
    /// ```
    pub fn add_input(&mut self, name: &str, tensor: Tensor) {
        self.runtime_inputs.insert(name.to_string(), tensor);
    }

    /// Builder-style method to add input (returns self for chaining)
    ///
    /// # Example
    /// ```
    /// let result = script
    ///     .with_input("x", tensor_x)
    ///     .with_input("y", tensor_y)
    ///     .run()?;
    /// ```
    pub fn with_input(mut self, name: &str, tensor: Tensor) -> Self {
        self.add_input(name, tensor);
        self
    }

    /// Clear all runtime inputs
    pub fn clear_inputs(&mut self) {
        self.runtime_inputs.clear();
    }

    /// Get current runtime inputs
    pub fn get_inputs(&self) -> &HashMap<String, Tensor> {
        &self.runtime_inputs
    }

    /// Compile the script for execution
    ///
    /// This method compiles the script IR into an optimized executable form.
    /// The compiled result is cached, so subsequent calls to `run()` will reuse it.
    ///
    /// # Example
    /// ```
    /// script.compile()?;  // Explicit compilation (optional)
    /// script.run()?;      // Uses cached compilation
    /// ```
    pub fn compile(&mut self) -> HoduResult<()> {
        // Validate that we have IR
        let _ir = self
            .ir
            .as_ref()
            .ok_or_else(|| HoduError::ScriptValidationFailed("Cannot compile script without IR".to_string()))?;

        // Determine target device
        let target_device = self.device.unwrap_or(Device::CPU);

        // Create executor based on backend preference
        let mut executor = self.create_executor(target_device)?;

        // Compile the script
        let compile_options = CompileOptions {
            target_device,
            ..Default::default()
        };

        let compiled_script = executor.compile(self, compile_options)?;

        // Cache the compiled script and executor
        self.compiled = Some(compiled_script);
        self.executor = Some(executor);

        Ok(())
    }

    /// Execute the script with previously added inputs
    ///
    /// If the script hasn't been compiled yet, it will be compiled automatically.
    /// Subsequent calls will reuse the cached compilation.
    ///
    /// # Returns
    /// * `HoduResult<ExecutionOutputs>` - Output tensors mapped by name
    ///
    /// # Example
    /// ```
    /// script.add_input("x", tensor_x);
    /// script.add_input("y", tensor_y);
    /// let outputs = script.run()?;
    /// let result = &outputs["result"];
    /// ```
    pub fn run(&mut self) -> HoduResult<ExecutionOutputs> {
        // Compile if not already compiled
        if self.compiled.is_none() || self.executor.is_none() {
            self.compile()?;
        }

        // Convert runtime inputs to ExecutionInputs format
        let inputs: HashMap<&str, Tensor> = self.runtime_inputs.iter().map(|(k, v)| (k.as_str(), *v)).collect();

        // Execute using cached compilation
        let executor = self.executor.as_ref().unwrap();
        let compiled = self.compiled.as_ref().unwrap();

        executor.execute(compiled, inputs)
    }

    /// Create appropriate executor based on backend and device settings
    fn create_executor(&self, target_device: Device) -> HoduResult<Executor> {
        use crate::backends::be_hodu::executor::HoduExecutor;
        #[cfg(feature = "xla")]
        use crate::backends::be_xla::executor::XlaExecutor;

        match self.backend.unwrap_or(Backend::HODU) {
            Backend::HODU => Ok(Executor::Hodu(HoduExecutor::new(target_device))),
            Backend::XLA => {
                #[cfg(feature = "xla")]
                {
                    Ok(Executor::Xla(XlaExecutor::new(target_device)?))
                }
                #[cfg(not(feature = "xla"))]
                {
                    Err(HoduError::UnsupportedBackend(Backend::XLA))
                }
            },
        }
    }

    #[cfg(feature = "serde")]
    /// Serialize the script to compressed bytes (works in also no-std)
    pub fn to_bytes(&self) -> HoduResult<Vec<u8>> {
        if let Some(ir) = &self.ir {
            let binary_data = bincode::encode_to_vec(ir, bincode::config::standard())
                .map_err(|e| HoduError::SerializationError(e.to_string()))?;

            #[cfg(feature = "std")]
            {
                use std::io::Write;
                let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
                encoder
                    .write_all(&binary_data)
                    .map_err(|e| HoduError::CompressionError(e.to_string()))?;
                let compressed_data = encoder
                    .finish()
                    .map_err(|e| HoduError::CompressionError(e.to_string()))?;
                Ok(compressed_data)
            }

            #[cfg(not(feature = "std"))]
            {
                // In no-std, return uncompressed data
                Ok(binary_data)
            }
        } else {
            Err(HoduError::ScriptValidationFailed(
                "Cannot serialize script without IR".to_string(),
            ))
        }
    }

    #[cfg(feature = "serde")]
    /// Deserialize the script from bytes (works in also no-std)
    pub fn from_bytes(data: &[u8]) -> HoduResult<Self> {
        #[cfg(feature = "std")]
        {
            // Try to decompress first (for std environment)
            match Self::from_compressed_bytes(data) {
                Ok(script) => Ok(script),
                Err(_) => {
                    // If decompression fails, try as uncompressed
                    let (ir, _): (ScriptIR, usize) = bincode::decode_from_slice(data, bincode::config::standard())
                        .map_err(|e| HoduError::SerializationError(e.to_string()))?;
                    Ok(Self::with_ir(ir))
                },
            }
        }

        #[cfg(not(feature = "std"))]
        {
            // In no-std, assume uncompressed data
            let (ir, _): (ScriptIR, usize) = bincode::decode_from_slice(data, bincode::config::standard())
                .map_err(|e| HoduError::SerializationError(e.to_string()))?;
            Ok(Self::with_ir(ir))
        }
    }

    #[cfg(all(feature = "serde", feature = "std"))]
    /// Helper function to handle compressed data (requires both serde and std)
    fn from_compressed_bytes(compressed_data: &[u8]) -> HoduResult<Self> {
        use std::io::Read;
        let mut decoder = flate2::read::GzDecoder::new(compressed_data);
        let mut binary_data = Vec::new();
        decoder
            .read_to_end(&mut binary_data)
            .map_err(|e| HoduError::DecompressionError(e.to_string()))?;

        let (ir, _): (ScriptIR, usize) = bincode::decode_from_slice(&binary_data, bincode::config::standard())
            .map_err(|e| HoduError::SerializationError(e.to_string()))?;

        Ok(Self::with_ir(ir))
    }

    #[cfg(all(feature = "serde", feature = "std"))]
    /// Save the script to a file (requires std feature)
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> HoduResult<()> {
        let compressed_data = self.to_bytes()?;

        let mut path = path.as_ref().to_path_buf();
        if path.extension().is_none() {
            path.set_extension("hoduscript");
        }

        std::fs::write(path, compressed_data).map_err(|e| HoduError::IoError(e.to_string()))?;
        Ok(())
    }

    #[cfg(all(feature = "serde", feature = "std"))]
    /// Load a script from a file (requires std feature)
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> HoduResult<Self> {
        let data = std::fs::read(path).map_err(|e| HoduError::IoError(e.to_string()))?;
        Self::from_bytes(&data)
    }
}
