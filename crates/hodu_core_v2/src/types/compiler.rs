use crate::types::device::Device;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    HODU,
    #[cfg(feature = "xla")]
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
            Backend::HODU => match device {
                Device::CPU => true,
                #[cfg(feature = "cuda")]
                Device::CUDA(_) => true,
                #[cfg(feature = "metal")]
                Device::Metal => true,
            },
            #[cfg(feature = "xla")]
            Backend::XLA => match device {
                Device::CPU => true,
                #[cfg(feature = "cuda")]
                Device::CUDA(_) => true,
                _ => false,
            },
        }
    }
}
