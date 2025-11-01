use crate::{layer::compat::*, types::Device};

// Use AtomicU8 for lock-free access to runtime device
// Device encoding: CPU=0, CUDA=1, Metal=2
static RUNTIME_DEVICE: AtomicU8 = AtomicU8::new(0); // Default: CPU

#[inline]
pub fn get_runtime_device() -> Device {
    let device_id = RUNTIME_DEVICE.load(Ordering::Relaxed);
    match device_id {
        0 => Device::CPU,
        #[cfg(feature = "cuda")]
        1 => Device::CUDA(0), // Default CUDA device 0
        #[cfg(feature = "metal")]
        2 => Device::Metal,
        _ => Device::CPU, // Fallback
    }
}

#[inline]
pub fn set_runtime_device(device: Device) {
    let device_id = match device {
        Device::CPU => 0,
        #[cfg(feature = "cuda")]
        Device::CUDA(_) => 1, // Store as CUDA (device index not preserved for runtime default)
        #[cfg(feature = "metal")]
        Device::Metal => 2,
    };
    RUNTIME_DEVICE.store(device_id, Ordering::Relaxed);
}
