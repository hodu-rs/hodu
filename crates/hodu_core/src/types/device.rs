use crate::layer::compat::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    #[default]
    CPU,
    #[cfg(feature = "cuda")]
    CUDA(usize),
    #[cfg(feature = "metal")]
    Metal,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::CPU => write!(f, "cpu"),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => write!(f, "cuda::{id}"),
            #[cfg(feature = "metal")]
            Device::Metal => write!(f, "Metal"),
        }
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Device[{}]", self)
    }
}

impl Device {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::CPU)
    }

    #[cfg(feature = "cuda")]
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::CUDA(_))
    }

    #[cfg(feature = "metal")]
    pub fn is_metal(&self) -> bool {
        matches!(self, Device::Metal)
    }
}
