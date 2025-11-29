//! C compiler toolchain abstraction
//!
//! Detects and invokes the appropriate C compiler for each platform.

use hodu_plugin::{HoduError, HoduResult};
use std::path::Path;
use std::process::Command;

/// Supported C compilers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CCompiler {
    /// Clang (default on macOS)
    Clang,
    /// GCC (default on Linux)
    Gcc,
    /// MSVC cl.exe (default on Windows)
    Msvc,
}

impl CCompiler {
    /// Get the default compiler for the current platform
    pub fn default_for_platform() -> Self {
        #[cfg(target_os = "macos")]
        {
            CCompiler::Clang
        }
        #[cfg(target_os = "linux")]
        {
            CCompiler::Gcc
        }
        #[cfg(target_os = "windows")]
        {
            CCompiler::Msvc
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            CCompiler::Gcc // fallback
        }
    }

    /// Get the compiler executable name
    pub fn executable(&self) -> &'static str {
        match self {
            CCompiler::Clang => "clang",
            CCompiler::Gcc => "gcc",
            CCompiler::Msvc => "cl.exe",
        }
    }

    /// Check if the compiler is available
    pub fn is_available(&self) -> bool {
        Command::new(self.executable())
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Get the shared library extension for the current platform
    pub fn shared_lib_extension() -> &'static str {
        #[cfg(target_os = "macos")]
        {
            "dylib"
        }
        #[cfg(target_os = "linux")]
        {
            "so"
        }
        #[cfg(target_os = "windows")]
        {
            "dll"
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            "so"
        }
    }

    /// Get the static library extension for the current platform
    pub fn static_lib_extension() -> &'static str {
        #[cfg(target_os = "windows")]
        {
            "lib"
        }
        #[cfg(not(target_os = "windows"))]
        {
            "a"
        }
    }

    /// Compile C source to shared library
    pub fn compile_to_shared_lib(
        &self,
        source_path: &Path,
        output_path: &Path,
        kernels_lib: &Path,
        opt_level: u8,
    ) -> HoduResult<()> {
        let mut cmd = Command::new(self.executable());
        let opt_flag = format!("-O{}", opt_level.min(3));

        // Check for BLAS library next to kernels lib
        let blas_lib = kernels_lib.with_file_name("libhodu_cpu_kernels_blas.a");

        match self {
            CCompiler::Clang | CCompiler::Gcc => {
                // Common flags for clang/gcc
                cmd.arg("-shared")
                    .arg("-fPIC")
                    .arg(&opt_flag)
                    .arg("-o")
                    .arg(output_path)
                    .arg(source_path)
                    .arg(kernels_lib);

                // Link BLAS library if it exists
                if blas_lib.exists() {
                    cmd.arg(&blas_lib);
                }

                // Platform-specific linker flags
                #[cfg(target_os = "macos")]
                {
                    cmd.arg("-dynamiclib");
                    // Link Accelerate framework for BLAS
                    cmd.arg("-framework").arg("Accelerate");
                }

                #[cfg(target_os = "linux")]
                {
                    cmd.arg("-lm"); // math library
                    cmd.arg("-lpthread"); // pthread library
                }
            }
            CCompiler::Msvc => {
                // MSVC flags
                cmd.arg("/LD") // Create DLL
                    .arg(format!("/O{}", opt_level.min(2))) // MSVC uses /O1 or /O2
                    .arg(format!("/Fe:{}", output_path.display()))
                    .arg(source_path)
                    .arg(kernels_lib);

                if blas_lib.exists() {
                    cmd.arg(&blas_lib);
                }
            }
        }

        let output = cmd
            .output()
            .map_err(|e| HoduError::BackendError(format!("Failed to run {}: {}", self.executable(), e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(HoduError::BackendError(format!(
                "{} compilation failed:\n{}",
                self.executable(),
                stderr
            )));
        }

        Ok(())
    }

    /// Compile C source to object file
    pub fn compile_to_object(
        &self,
        source_path: &Path,
        output_path: &Path,
        include_dirs: &[&Path],
    ) -> HoduResult<()> {
        let mut cmd = Command::new(self.executable());

        match self {
            CCompiler::Clang | CCompiler::Gcc => {
                cmd.arg("-c")
                    .arg("-fPIC")
                    .arg("-O2")
                    .arg("-o")
                    .arg(output_path)
                    .arg(source_path);

                for inc in include_dirs {
                    cmd.arg("-I").arg(inc);
                }
            }
            CCompiler::Msvc => {
                cmd.arg("/c")
                    .arg("/O2")
                    .arg(format!("/Fo:{}", output_path.display()))
                    .arg(source_path);

                for inc in include_dirs {
                    cmd.arg(format!("/I{}", inc.display()));
                }
            }
        }

        let output = cmd
            .output()
            .map_err(|e| HoduError::BackendError(format!("Failed to run {}: {}", self.executable(), e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(HoduError::BackendError(format!(
                "{} compilation failed:\n{}",
                self.executable(),
                stderr
            )));
        }

        Ok(())
    }
}

/// Find the best available compiler
pub fn find_compiler() -> HoduResult<CCompiler> {
    let default = CCompiler::default_for_platform();

    if default.is_available() {
        return Ok(default);
    }

    // Try alternatives
    let alternatives = match default {
        CCompiler::Clang => vec![CCompiler::Gcc],
        CCompiler::Gcc => vec![CCompiler::Clang],
        CCompiler::Msvc => vec![CCompiler::Gcc, CCompiler::Clang],
    };

    for alt in alternatives {
        if alt.is_available() {
            return Ok(alt);
        }
    }

    Err(HoduError::BackendError(format!(
        "No C compiler found. Please install {} or an alternative.",
        default.executable()
    )))
}
