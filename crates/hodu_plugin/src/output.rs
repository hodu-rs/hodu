//! Output format definitions for AOT compilation

use hodu_compat::*;

/// Output format for AOT compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum OutputFormat {
    // === Hodu ===
    /// Hodu Snapshot (.hdss) - serialized computation graph for interpreter
    HoduSnapshot,

    // === CPU (Native) ===
    /// Object file (.o)
    Object,
    /// Shared library (.so / .dylib / .dll)
    SharedLib,
    /// Static library (.a / .lib)
    StaticLib,
    /// Executable binary
    Executable,

    // === LLVM IR (debugging) ===
    /// LLVM IR text (.ll)
    LlvmIR,
    /// LLVM bitcode (.bc)
    LlvmBitcode,
    /// Assembly (.s)
    Assembly,

    // === CUDA ===
    /// PTX text (.ptx)
    Ptx,
    /// CUDA binary (.cubin)
    Cubin,
    /// Fat binary for multiple architectures (.fatbin)
    Fatbin,

    // === Metal ===
    /// Metal Shading Language source (.metal)
    Msl,
    /// Apple IR (.air)
    Air,
    /// Metal library (.metallib)
    Metallib,

    // === ROCm ===
    /// AMD GPU binary (.hsaco)
    Hsaco,

    // === Portable ===
    /// SPIR-V for Vulkan/OpenCL (.spv)
    SpirV,
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputFormat::HoduSnapshot => write!(f, "hdss"),
            OutputFormat::Object => write!(f, "object"),
            OutputFormat::SharedLib => write!(f, "shared"),
            OutputFormat::StaticLib => write!(f, "static"),
            OutputFormat::Executable => write!(f, "executable"),
            OutputFormat::LlvmIR => write!(f, "llvm-ir"),
            OutputFormat::LlvmBitcode => write!(f, "llvm-bc"),
            OutputFormat::Assembly => write!(f, "asm"),
            OutputFormat::Ptx => write!(f, "ptx"),
            OutputFormat::Cubin => write!(f, "cubin"),
            OutputFormat::Fatbin => write!(f, "fatbin"),
            OutputFormat::Msl => write!(f, "msl"),
            OutputFormat::Air => write!(f, "air"),
            OutputFormat::Metallib => write!(f, "metallib"),
            OutputFormat::Hsaco => write!(f, "hsaco"),
            OutputFormat::SpirV => write!(f, "spirv"),
        }
    }
}

impl OutputFormat {
    /// Parse format from string (e.g., "shared", "ptx", "metallib")
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "hdss" | "snapshot" | "hodu-snapshot" => Some(OutputFormat::HoduSnapshot),
            "object" | "obj" => Some(OutputFormat::Object),
            "shared" | "so" | "dylib" | "dll" => Some(OutputFormat::SharedLib),
            "static" | "a" | "lib" => Some(OutputFormat::StaticLib),
            "executable" | "exe" | "bin" => Some(OutputFormat::Executable),
            "llvm-ir" | "ll" => Some(OutputFormat::LlvmIR),
            "llvm-bc" | "bc" => Some(OutputFormat::LlvmBitcode),
            "asm" | "assembly" | "s" => Some(OutputFormat::Assembly),
            "ptx" => Some(OutputFormat::Ptx),
            "cubin" => Some(OutputFormat::Cubin),
            "fatbin" => Some(OutputFormat::Fatbin),
            "msl" | "metal" => Some(OutputFormat::Msl),
            "air" => Some(OutputFormat::Air),
            "metallib" => Some(OutputFormat::Metallib),
            "hsaco" => Some(OutputFormat::Hsaco),
            "spirv" | "spv" => Some(OutputFormat::SpirV),
            _ => None,
        }
    }

    /// Infer format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "hdss" => Some(OutputFormat::HoduSnapshot),
            "o" => Some(OutputFormat::Object),
            "so" | "dylib" | "dll" => Some(OutputFormat::SharedLib),
            "a" | "lib" => Some(OutputFormat::StaticLib),
            "ll" => Some(OutputFormat::LlvmIR),
            "bc" => Some(OutputFormat::LlvmBitcode),
            "s" => Some(OutputFormat::Assembly),
            "ptx" => Some(OutputFormat::Ptx),
            "cubin" => Some(OutputFormat::Cubin),
            "fatbin" => Some(OutputFormat::Fatbin),
            "metal" => Some(OutputFormat::Msl),
            "air" => Some(OutputFormat::Air),
            "metallib" => Some(OutputFormat::Metallib),
            "hsaco" => Some(OutputFormat::Hsaco),
            "spv" => Some(OutputFormat::SpirV),
            _ => None,
        }
    }

    /// Get the default file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            OutputFormat::HoduSnapshot => "hdss",
            OutputFormat::Object => "o",
            OutputFormat::SharedLib => {
                #[cfg(target_os = "linux")]
                {
                    "so"
                }
                #[cfg(target_os = "macos")]
                {
                    "dylib"
                }
                #[cfg(target_os = "windows")]
                {
                    "dll"
                }
                #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
                {
                    "so"
                }
            },
            OutputFormat::StaticLib => {
                #[cfg(any(target_os = "linux", target_os = "macos"))]
                {
                    "a"
                }
                #[cfg(target_os = "windows")]
                {
                    "lib"
                }
                #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
                {
                    "a"
                }
            },
            OutputFormat::Executable => "",
            OutputFormat::LlvmIR => "ll",
            OutputFormat::LlvmBitcode => "bc",
            OutputFormat::Assembly => "s",
            OutputFormat::Ptx => "ptx",
            OutputFormat::Cubin => "cubin",
            OutputFormat::Fatbin => "fatbin",
            OutputFormat::Msl => "metal",
            OutputFormat::Air => "air",
            OutputFormat::Metallib => "metallib",
            OutputFormat::Hsaco => "hsaco",
            OutputFormat::SpirV => "spv",
        }
    }
}
