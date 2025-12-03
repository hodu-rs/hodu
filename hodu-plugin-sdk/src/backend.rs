//! Backend plugin types
//!
//! Types for backend plugins that execute models on various devices.

/// Target device for plugin execution
///
/// Using String for extensibility - plugins can define custom devices.
/// Convention: lowercase with `::` separator for device index.
/// Common values: "cpu", "cuda::0", "metal", "vulkan", "webgpu", "rocm::0"
pub type Device = String;

/// Parse device ID from device string (e.g., "cuda::0" -> 0)
pub fn parse_device_id(device: &str) -> Option<usize> {
    device.split("::").nth(1).and_then(|id| id.parse().ok())
}

/// Get device type from device string (e.g., "cuda::0" -> "cuda")
pub fn device_type(device: &str) -> &str {
    device.split("::").next().unwrap_or(device)
}

/// Build target specification for AOT compilation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BuildTarget {
    /// Target triple (e.g., "x86_64-unknown-linux-gnu", "aarch64-apple-darwin")
    pub triple: String,
    /// Target device (e.g., "cpu", "metal", "cuda::0")
    pub device: String,
}

impl BuildTarget {
    /// Create a new build target
    pub fn new(triple: impl Into<String>, device: impl Into<String>) -> Self {
        Self {
            triple: triple.into(),
            device: device.into(),
        }
    }

    /// Create a build target for the current host system
    pub fn host(device: impl Into<String>) -> Self {
        Self::new(env!("HOST_TARGET"), device)
    }
}
