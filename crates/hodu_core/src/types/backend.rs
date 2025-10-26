use crate::types::device::Device;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    HODU,
    XLA,
}

impl Default for Backend {
    fn default() -> Self {
        Self::HODU
    }
}

impl Backend {
    pub fn is_supported(&self, device: Device) -> bool {
        match self {
            Backend::HODU => {
                matches!(device, Device::CPU | Device::CUDA(_) | Device::Metal)
            },
            Backend::XLA => {
                matches!(device, Device::CPU | Device::CUDA(_))
            },
        }
    }
}
