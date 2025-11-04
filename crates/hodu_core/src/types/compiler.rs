use crate::{layer::compat::*, types::device::Device};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Compiler {
    HODU,
    #[cfg(feature = "xla")]
    XLA,
}

impl Default for Compiler {
    fn default() -> Self {
        Self::HODU
    }
}

impl fmt::Display for Compiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Compiler::HODU => write!(f, "hodu"),
            #[cfg(feature = "xla")]
            Compiler::XLA => write!(f, "xla"),
        }
    }
}

impl fmt::Debug for Compiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Compiler[{}]", self)
    }
}

impl Compiler {
    pub fn is_supported(&self, device: Device) -> bool {
        match self {
            Compiler::HODU => match device {
                Device::CPU => true,
                #[cfg(feature = "cuda")]
                Device::CUDA(_) => true,
                #[cfg(feature = "metal")]
                Device::Metal => true,
            },
            #[cfg(feature = "xla")]
            Compiler::XLA => match device {
                Device::CPU => true,
                #[cfg(feature = "cuda")]
                Device::CUDA(_) => true,
                #[cfg(feature = "metal")]
                Device::Metal => false,
            },
        }
    }
}
