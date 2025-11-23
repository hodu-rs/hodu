use crate::{
    compat::*,
    types::{Device, Runtime},
};

/// Target architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetArch {
    X86_64,
    Aarch64,
    Arm,
}

impl fmt::Display for TargetArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::X86_64 => write!(f, "x86_64"),
            Self::Aarch64 => write!(f, "aarch64"),
            Self::Arm => write!(f, "arm"),
        }
    }
}

/// Target vendor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetVendor {
    Apple,
    Pc,
    Unknown,
}

impl fmt::Display for TargetVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Apple => write!(f, "apple"),
            Self::Pc => write!(f, "pc"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Target OS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetOS {
    Darwin,
    Linux,
    Windows,
    None,
}

impl fmt::Display for TargetOS {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Darwin => write!(f, "darwin"),
            Self::Linux => write!(f, "linux"),
            Self::Windows => write!(f, "windows"),
            Self::None => write!(f, "none"),
        }
    }
}

/// Target environment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetEnv {
    Gnu,
    Msvc,
    Musl,
}

impl fmt::Display for TargetEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gnu => write!(f, "gnu"),
            Self::Msvc => write!(f, "msvc"),
            Self::Musl => write!(f, "musl"),
        }
    }
}

/// Execution environment configuration
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Target device for execution
    pub device: Device,
    /// Runtime backend
    pub runtime: Runtime,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            device: Device::CPU,
            runtime: Runtime::HODU,
        }
    }
}

impl ExecutionConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn runtime(mut self, runtime: Runtime) -> Self {
        self.runtime = runtime;
        self
    }
}

/// Build target configuration
#[derive(Debug, Clone)]
pub struct TargetConfig {
    /// Target architecture
    pub arch: TargetArch,
    /// Target vendor
    pub vendor: TargetVendor,
    /// Target OS
    pub os: TargetOS,
    /// Target environment (optional)
    pub env: Option<TargetEnv>,
}

impl Default for TargetConfig {
    fn default() -> Self {
        // Detect current platform at compile time
        const ARCH: TargetArch = {
            #[cfg(target_arch = "x86_64")]
            {
                TargetArch::X86_64
            }
            #[cfg(target_arch = "aarch64")]
            {
                TargetArch::Aarch64
            }
            #[cfg(target_arch = "arm")]
            {
                TargetArch::Arm
            }
        };

        const VENDOR: TargetVendor = {
            #[cfg(target_vendor = "apple")]
            {
                TargetVendor::Apple
            }
            #[cfg(target_vendor = "pc")]
            {
                TargetVendor::Pc
            }
            #[cfg(target_vendor = "unknown")]
            {
                TargetVendor::Unknown
            }
        };

        const OS: TargetOS = {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                TargetOS::Darwin
            }
            #[cfg(target_os = "linux")]
            {
                TargetOS::Linux
            }
            #[cfg(target_os = "windows")]
            {
                TargetOS::Windows
            }
            #[cfg(target_os = "none")]
            {
                TargetOS::None
            }
        };

        const ENV: Option<TargetEnv> = {
            #[cfg(target_env = "gnu")]
            {
                Some(TargetEnv::Gnu)
            }
            #[cfg(target_env = "msvc")]
            {
                Some(TargetEnv::Msvc)
            }
            #[cfg(target_env = "musl")]
            {
                Some(TargetEnv::Musl)
            }
            #[cfg(target_env = "")]
            {
                None
            }
        };

        Self {
            arch: ARCH,
            vendor: VENDOR,
            os: OS,
            env: ENV,
        }
    }
}

impl TargetConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn arch(mut self, arch: TargetArch) -> Self {
        self.arch = arch;
        self
    }

    pub fn vendor(mut self, vendor: TargetVendor) -> Self {
        self.vendor = vendor;
        self
    }

    pub fn os(mut self, os: TargetOS) -> Self {
        self.os = os;
        self
    }

    pub fn env(mut self, env: Option<TargetEnv>) -> Self {
        self.env = env;
        self
    }

    /// Get target triple string (e.g., "x86_64-apple-darwin")
    pub fn triple(&self) -> String {
        if let Some(env) = self.env {
            format!("{}-{}-{}-{}", self.arch, self.vendor, self.os, env)
        } else {
            format!("{}-{}-{}", self.arch, self.vendor, self.os)
        }
    }
}

/// Build configuration
#[derive(Default, Debug, Clone)]
pub struct BuildConfig {
    /// Execution environment
    pub execution: ExecutionConfig,
    /// Build target
    pub target: TargetConfig,
}

impl BuildConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn execution(mut self, execution: ExecutionConfig) -> Self {
        self.execution = execution;
        self
    }

    pub fn target(mut self, target: TargetConfig) -> Self {
        self.target = target;
        self
    }
}
