pub mod builder;
pub mod capture;
pub mod snapshot;

use crate::{
    compat::*,
    error::{HoduError, HoduResult},
    tensor::Tensor,
    types::{Device, Runtime},
};
pub use builder::{BuildConfig, BuildType, Builder, TargetArch, TargetConfig, TargetEnv, TargetOS, TargetVendor};
pub use capture::{CaptureBoard, CaptureBoardId, CapturedInput, CapturedOp, CapturedTarget};
pub use snapshot::{Snapshot, SnapshotConstant, SnapshotInput, SnapshotNode, SnapshotTarget, SnapshotTensorId};

/// Script holds the Hodu Script IR and provides compilation/execution interface
pub struct Script {
    snapshot: Snapshot,
    compiled: Option<()>, // TODO: Store actual compiled state
    device: Option<Device>,
    runtime: Option<Runtime>,
}

impl Script {
    /// Create a new Script from a Snapshot
    pub fn new(snapshot: Snapshot) -> Self {
        Self {
            snapshot,
            compiled: None,
            device: None,
            runtime: None,
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

    /// Get device
    pub fn device(&self) -> Option<Device> {
        self.device
    }

    /// Get runtime
    pub fn runtime(&self) -> Option<Runtime> {
        self.runtime
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
    /// This prepares the script for JIT execution by generating LLVM IR
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

        // TODO: Generate and cache LLVM JIT engine or XLA executable

        self.compiled = Some(());
        Ok(())
    }

    /// Run the script with inputs and return outputs as HashMap<target_name, Tensor>
    /// Automatically compiles if not already compiled
    pub fn run(&mut self, _inputs: &[(&str, &Tensor)]) -> HoduResult<HashMap<String, Tensor>> {
        // Auto-compile if needed
        if self.compiled.is_none() {
            self.compile()?;
        }

        // TODO: Execute using the compiled runtime and return outputs
        // TODO: Validate inputs match snapshot.inputs
        // For now, return empty HashMap
        Err(HoduError::UnsupportedOperation(
            "JIT execution not yet implemented".into(),
        ))
    }
}
