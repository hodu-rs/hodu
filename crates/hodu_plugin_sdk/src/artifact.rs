//! Compiled artifact types for AOT compilation output

use hodu_core::types::Device;

/// Build output format for AOT compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuildFormat {
    // === CPU (Native) ===
    /// Object file (.o)
    Object,
    /// Shared library (.so / .dylib / .dll)
    SharedLib,
    /// Static library (.a / .lib)
    StaticLib,
    /// Executable binary
    Executable,

    // === LLVM IR ===
    /// LLVM IR text (.ll)
    LLVMIR,
    /// LLVM Bitcode (.bc)
    LLVMBitcode,

    // === Metal ===
    /// Metal library (.metallib)
    MetalLib,
    /// Metal IR (.metalir)
    MetalIR,

    // === CUDA ===
    /// PTX assembly (.ptx)
    PTX,
    /// CUDA binary (.cubin)
    CUBIN,

    // === WebGPU ===
    /// WGSL shader (.wgsl)
    WGSL,
    /// SPIR-V binary (.spv)
    SPIRV,
}

impl BuildFormat {
    /// Get the typical file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Object => "o",
            Self::SharedLib => {
                #[cfg(target_os = "macos")]
                {
                    "dylib"
                }
                #[cfg(target_os = "windows")]
                {
                    "dll"
                }
                #[cfg(not(any(target_os = "macos", target_os = "windows")))]
                {
                    "so"
                }
            },
            Self::StaticLib => {
                #[cfg(target_os = "windows")]
                {
                    "lib"
                }
                #[cfg(not(target_os = "windows"))]
                {
                    "a"
                }
            },
            Self::Executable => {
                #[cfg(target_os = "windows")]
                {
                    "exe"
                }
                #[cfg(not(target_os = "windows"))]
                {
                    ""
                }
            },
            Self::LLVMIR => "ll",
            Self::LLVMBitcode => "bc",
            Self::MetalLib => "metallib",
            Self::MetalIR => "metalir",
            Self::PTX => "ptx",
            Self::CUBIN => "cubin",
            Self::WGSL => "wgsl",
            Self::SPIRV => "spv",
        }
    }
}

/// Compiled artifact produced by a backend's build function
#[derive(Debug, Clone)]
pub struct CompiledArtifact {
    /// The format of this artifact
    pub format: BuildFormat,

    /// Target device this was compiled for
    pub device: Device,

    /// The compiled binary data
    pub data: Vec<u8>,

    /// Metadata about inputs
    pub inputs: Vec<ArtifactTensorInfo>,

    /// Metadata about outputs
    pub outputs: Vec<ArtifactTensorInfo>,

    /// Optional symbol table for native artifacts
    pub symbols: Option<ArtifactSymbols>,
}

/// Tensor metadata in compiled artifact
#[derive(Debug, Clone)]
pub struct ArtifactTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: ArtifactDType,
}

/// Data type in artifact (fixed enum for ABI stability)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ArtifactDType {
    Bool = 0,
    F8E4M3 = 1,
    F8E5M2 = 2,
    BF16 = 3,
    F16 = 4,
    F32 = 5,
    F64 = 6,
    U8 = 7,
    U16 = 8,
    U32 = 9,
    U64 = 10,
    I8 = 11,
    I16 = 12,
    I32 = 13,
    I64 = 14,
}

impl From<hodu_core::types::DType> for ArtifactDType {
    fn from(dtype: hodu_core::types::DType) -> Self {
        use hodu_core::types::DType;
        match dtype {
            DType::BOOL => ArtifactDType::Bool,
            DType::F8E4M3 => ArtifactDType::F8E4M3,
            DType::F8E5M2 => ArtifactDType::F8E5M2,
            DType::BF16 => ArtifactDType::BF16,
            DType::F16 => ArtifactDType::F16,
            DType::F32 => ArtifactDType::F32,
            DType::F64 => ArtifactDType::F64,
            DType::U8 => ArtifactDType::U8,
            DType::U16 => ArtifactDType::U16,
            DType::U32 => ArtifactDType::U32,
            DType::U64 => ArtifactDType::U64,
            DType::I8 => ArtifactDType::I8,
            DType::I16 => ArtifactDType::I16,
            DType::I32 => ArtifactDType::I32,
            DType::I64 => ArtifactDType::I64,
        }
    }
}

/// Symbol information for native artifacts
#[derive(Debug, Clone, Default)]
pub struct ArtifactSymbols {
    /// Entry point function name
    pub entry_point: String,
    /// Additional exported symbols
    pub exports: Vec<String>,
}

impl CompiledArtifact {
    /// Create a new compiled artifact
    pub fn new(format: BuildFormat, device: Device, data: Vec<u8>) -> Self {
        Self {
            format,
            device,
            data,
            inputs: Vec::new(),
            outputs: Vec::new(),
            symbols: None,
        }
    }

    /// Add input tensor info
    pub fn with_input(mut self, name: impl Into<String>, shape: Vec<usize>, dtype: ArtifactDType) -> Self {
        self.inputs.push(ArtifactTensorInfo {
            name: name.into(),
            shape,
            dtype,
        });
        self
    }

    /// Add output tensor info
    pub fn with_output(mut self, name: impl Into<String>, shape: Vec<usize>, dtype: ArtifactDType) -> Self {
        self.outputs.push(ArtifactTensorInfo {
            name: name.into(),
            shape,
            dtype,
        });
        self
    }

    /// Set symbol information
    pub fn with_symbols(mut self, symbols: ArtifactSymbols) -> Self {
        self.symbols = Some(symbols);
        self
    }

    /// Get the raw data
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get data size in bytes
    pub fn size(&self) -> usize {
        self.data.len()
    }
}
