use crate::compat::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    CUDA(usize),
    METAL,
}

impl Default for Device {
    fn default() -> Self {
        Self::CPU
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::CPU => write!(f, "cpu"),
            Device::CUDA(id) => write!(f, "cuda::{id}"),
            Device::METAL => write!(f, "metal"),
        }
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Device {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::CPU)
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::CUDA(_))
    }

    pub fn is_metal(&self) -> bool {
        matches!(self, Device::METAL)
    }
}
