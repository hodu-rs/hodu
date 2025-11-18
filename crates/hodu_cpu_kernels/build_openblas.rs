#![allow(dead_code)]

pub struct OpenBlasConfig {
    pub available: bool,
    pub include_path: Option<String>,
    pub lib_path: Option<String>,
}

impl OpenBlasConfig {
    pub fn detect() -> Self {
        if let Some((include_path, lib_path)) = detect_openblas_paths() {
            return Self {
                available: true,
                include_path: if include_path.is_empty() {
                    None
                } else {
                    Some(include_path)
                },
                lib_path,
            };
        }

        Self::none()
    }

    fn none() -> Self {
        Self {
            available: false,
            include_path: None,
            lib_path: None,
        }
    }

    pub fn apply_to_build(&self, build: &mut cc::Build) {
        if !self.available {
            return;
        }
        build.define("USE_BLAS", None);
        if let Some(ref path) = self.include_path {
            build.include(path);
        }
    }

    pub fn warn_if_not_found(&self) {
        if !self.available {
            println!("cargo:warning=OpenBLAS feature enabled but library not found");
            println!("cargo:warning=Install: brew install openblas (macOS) or apt install libopenblas-dev (Linux)");
        }
    }

    pub fn setup_linking(&self) {
        if !self.available {
            return;
        }

        // Library search path
        if let Ok(lib_dir) = std::env::var("OPENBLAS_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", lib_dir);
        } else if let Ok(dir) = std::env::var("OPENBLAS_DIR") {
            println!("cargo:rustc-link-search=native={}/lib", dir);
        } else if let Some(ref path) = self.lib_path {
            println!("cargo:rustc-link-search=native={}", path);
        }

        // Link OpenBLAS
        println!("cargo:rustc-link-lib=openblas");

        // Platform dependencies
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

        match target_os.as_str() {
            "linux" => {
                println!("cargo:rustc-link-lib=pthread");
                if target_env != "musl" {
                    println!("cargo:rustc-link-lib=gfortran");
                }
            },
            "windows" => {
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

    // Environment variables
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

    // Skip auto-detection for cross-compilation
    if is_cross_compile() {
        return None;
    }

    let mut include_path = None;
    let mut lib_path = None;

    // Platform-specific detection
    if cfg!(target_os = "macos") {
        if let Ok(output) = Command::new("brew").args(["--prefix", "openblas"]).output() {
            if output.status.success() {
                let prefix = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let inc = std::path::Path::new(&prefix).join("include");
                let lib = std::path::Path::new(&prefix).join("lib");
                if inc.join("cblas.h").exists() {
                    include_path = Some(inc.to_string_lossy().to_string());
                    lib_path = Some(lib.to_string_lossy().to_string());
                }
            }
        }
    } else if cfg!(target_os = "linux") {
        // Try pkg-config
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

        // Fallback: standard paths
        if include_path.is_none() {
            for path in ["/usr/include", "/usr/local/include", "/usr/include/openblas"] {
                if std::path::Path::new(path).join("cblas.h").exists() {
                    include_path = Some(path.to_string());
                    lib_path = Some("/usr/lib".to_string());
                    break;
                }
            }
        }
    }

    // Test compilation
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

    let mut file = fs::File::create(&test_c).ok()?;
    file.write_all(test_code.as_bytes()).ok()?;
    drop(file);

    let mut build = cc::Build::new();
    build.cargo_warnings(false);
    if let Some(ref path) = include_path {
        build.include(path);
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        build.file(&test_c).try_compile("test_openblas")
    }));

    let _ = fs::remove_file(&test_c);

    match result {
        Ok(Ok(_)) => Some((include_path.unwrap_or_default(), lib_path)),
        _ => None,
    }
}

fn is_cross_compile() -> bool {
    let target = std::env::var("TARGET").unwrap_or_default();
    let host = std::env::var("HOST").unwrap_or_default();
    target != host
}
