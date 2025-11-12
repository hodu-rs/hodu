pub struct OpenBlasConfig {
    pub available: bool,
    pub include_path: Option<String>,
    pub lib_path: Option<String>,
}

impl OpenBlasConfig {
    pub fn detect() -> Self {
        // Skip if explicitly disabled
        if std::env::var("HODU_DISABLE_BLAS").is_ok() {
            return OpenBlasConfig {
                available: false,
                include_path: None,
                lib_path: None,
            };
        }

        if let Some((include_path, lib_path)) = detect_openblas_paths() {
            OpenBlasConfig {
                available: true,
                include_path: if include_path.is_empty() {
                    None
                } else {
                    Some(include_path)
                },
                lib_path,
            }
        } else {
            OpenBlasConfig {
                available: false,
                include_path: None,
                lib_path: None,
            }
        }
    }

    pub fn apply_to_build(&self, build: &mut cc::Build) {
        if self.available {
            build.define("USE_BLAS", None);

            if let Some(ref path) = self.include_path {
                build.include(path);
            }
        }
    }

    pub fn print_status(&self) {
        if !self.available {
            println!("cargo:warning=OpenBLAS not found, building without BLAS acceleration");
            println!("cargo:warning=Install OpenBLAS for better performance:");
            println!("cargo:warning=  macOS: brew install openblas gfortran");
            println!("cargo:warning=  Linux: sudo apt install libopenblas-dev pkg-config gfortran");
            println!("cargo:warning=  Windows: install via vcpkg or MinGW");
        }
    }

    pub fn setup_linking(&self) {
        if !self.available {
            return;
        }

        // Add library search path if specified (for cross-compilation)
        if let Ok(lib_dir) = std::env::var("OPENBLAS_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", lib_dir);
        } else if let Ok(dir) = std::env::var("OPENBLAS_DIR") {
            println!("cargo:rustc-link-search=native={}/lib", dir);
        } else if let Some(ref lib_path) = self.lib_path {
            println!("cargo:rustc-link-search=native={}", lib_path);
        }

        // Link OpenBLAS
        println!("cargo:rustc-link-lib=openblas");

        // Platform-specific system libraries for OpenBLAS
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
        let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

        match target_os.as_str() {
            "macos" => {
                // macOS: link Accelerate framework
                println!("cargo:rustc-link-lib=framework=Accelerate");
            },
            "linux" => {
                // Linux: link pthread
                println!("cargo:rustc-link-lib=pthread");
                if target_env != "musl" {
                    println!("cargo:rustc-link-lib=gfortran");
                }
            },
            "windows" => {
                // Windows: OpenBLAS dependencies
                println!("cargo:rustc-link-lib=gfortran");
                println!("cargo:rustc-link-lib=quadmath");
                println!("cargo:rustc-link-lib=gcc");
            },
            _ => {
                println!("cargo:rustc-link-lib=pthread");
            },
        }
    }
}

fn detect_openblas_paths() -> Option<(String, Option<String>)> {
    use std::fs;
    use std::io::Write;
    use std::process::Command;

    // Strategy 1: Check environment variables (for cross-compilation)
    if let Ok(include_dir) = std::env::var("OPENBLAS_INCLUDE_DIR") {
        let lib_dir = std::env::var("OPENBLAS_LIB_DIR").ok();
        return Some((include_dir, lib_dir));
    }

    if let Ok(dir) = std::env::var("OPENBLAS_DIR") {
        let include_dir = std::path::Path::new(&dir).join("include");
        let lib_dir = std::path::Path::new(&dir).join("lib");
        return Some((
            include_dir.to_string_lossy().to_string(),
            Some(lib_dir.to_string_lossy().to_string()),
        ));
    }

    // Create a temporary test file
    let test_code = r#"
#include <cblas.h>
int main() {
    float a = 1.0f, b = 2.0f;
    cblas_sdot(1, &a, 1, &b, 1);
    return 0;
}
"#;

    let out_dir = std::env::var("OUT_DIR").ok()?;
    let test_c = std::path::Path::new(&out_dir).join("test_openblas.c");

    // Write test file
    if let Ok(mut file) = fs::File::create(&test_c) {
        let _ = file.write_all(test_code.as_bytes());
    } else {
        return None;
    }

    // Strategy 2: Auto-detect OpenBLAS (host compilation only)
    let mut include_path = None;
    let mut lib_path = None;
    let is_cross_compiling = is_cross_compile();

    if !is_cross_compiling {
        // Try package manager specific detection
        if cfg!(target_os = "macos") {
            // macOS: try Homebrew
            if let Ok(output) = Command::new("brew").args(["--prefix", "openblas"]).output() {
                if output.status.success() {
                    let prefix = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    let inc_dir = std::path::Path::new(&prefix).join("include");
                    let lib_dir = std::path::Path::new(&prefix).join("lib");
                    if inc_dir.join("cblas.h").exists() {
                        include_path = Some(inc_dir.to_string_lossy().to_string());
                        lib_path = Some(lib_dir.to_string_lossy().to_string());
                    }
                }
            }
        } else if cfg!(target_os = "linux") {
            // Linux: try pkg-config first
            if let Ok(output) = Command::new("pkg-config")
                .args(["--cflags-only-I", "openblas"])
                .output()
            {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    for flag in stdout.split_whitespace() {
                        if let Some(path) = flag.strip_prefix("-I") {
                            if std::path::Path::new(path).join("cblas.h").exists() {
                                include_path = Some(path.to_string());
                                break;
                            }
                        }
                    }
                }
            }

            // Try to get lib path from pkg-config too
            if include_path.is_some() {
                if let Ok(output) = Command::new("pkg-config").args(["--libs-only-L", "openblas"]).output() {
                    if output.status.success() {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        for flag in stdout.split_whitespace() {
                            if let Some(path) = flag.strip_prefix("-L") {
                                lib_path = Some(path.to_string());
                                break;
                            }
                        }
                    }
                }
            }

            // Fallback: check standard locations if pkg-config failed
            if include_path.is_none() {
                let standard_paths = [
                    "/usr/include",
                    "/usr/local/include",
                    "/usr/include/openblas",
                    "/usr/local/include/openblas",
                ];
                for path in &standard_paths {
                    if std::path::Path::new(path).join("cblas.h").exists() {
                        include_path = Some(path.to_string());
                        // Standard lib paths
                        lib_path = Some("/usr/lib".to_string());
                        break;
                    }
                }
            }
        }
        // Windows: rely on cc crate's default search paths (vcpkg, etc.)
    }

    // Try to compile with found include path
    let mut build = cc::Build::new();
    build.cargo_warnings(false);

    if let Some(ref path) = include_path {
        build.include(path);
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        build.file(&test_c).try_compile("test_openblas")
    }));

    // Clean up
    let _ = fs::remove_file(&test_c);

    match result {
        Ok(Ok(_)) => {
            if let Some(inc_path) = include_path {
                Some((inc_path, lib_path))
            } else {
                Some((String::new(), lib_path))
            }
        },
        _ => None,
    }
}

fn is_cross_compile() -> bool {
    let target = std::env::var("TARGET").unwrap_or_default();
    let host = std::env::var("HOST").unwrap_or_default();
    target != host
}
