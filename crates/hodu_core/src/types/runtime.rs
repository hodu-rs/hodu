use crate::{compat::*, types::device::Device};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Runtime {
    #[default]
    HODU,
    #[cfg(feature = "xla")]
    XLA,
}

impl fmt::Display for Runtime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Runtime::HODU => write!(f, "hodu"),
            #[cfg(feature = "xla")]
            Runtime::XLA => write!(f, "xla"),
        }
    }
}

impl fmt::Debug for Runtime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Runtime[{}]", self)
    }
}

impl Runtime {
    pub fn is_supported(&self, device: Device) -> bool {
        match self {
            Runtime::HODU => match device {
                Device::CPU => true,
                #[cfg(feature = "cuda")]
                Device::CUDA(_) => true,
                #[cfg(feature = "metal")]
                Device::Metal => true,
            },
            #[cfg(feature = "xla")]
            Runtime::XLA => match device {
                Device::CPU => true,
                #[cfg(feature = "cuda")]
                Device::CUDA(_) => true,
                #[cfg(feature = "metal")]
                Device::Metal => false,
            },
        }
    }
}
