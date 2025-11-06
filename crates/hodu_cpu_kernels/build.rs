fn main() {
    use std::process::Command;

    // Build OpenBLAS if not already built
    let openblas_dir = std::path::PathBuf::from("libs/OpenBLAS");
    let openblas_lib = openblas_dir.join("libopenblas.a");

    if openblas_dir.exists() && !openblas_lib.exists() {
        println!("cargo:warning=Building OpenBLAS...");

        // Detect target architecture and OS
        let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
        let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

        // Determine OpenBLAS TARGET based on architecture
        let openblas_target = match target_arch.as_str() {
            "aarch64" => "ARMV8",
            "arm" => "ARMV7",
            "x86_64" => {
                // Use HASWELL for modern x86_64, GENERIC for older
                if target_env == "musl" {
                    "GENERIC"
                } else {
                    "HASWELL"
                }
            },
            "x86" => "GENERIC",
            "riscv64" => "RISCV64_GENERIC",
            _ => "GENERIC",
        };

        // Determine compiler and flags based on OS
        let (make_cmd, cc_env) = match target_os.as_str() {
            "windows" => {
                // Windows: use mingw32-make or make with MSYS2
                if Command::new("mingw32-make").arg("--version").output().is_ok() {
                    ("mingw32-make", "CC=gcc")
                } else {
                    ("make", "CC=gcc")
                }
            },
            _ => ("make", ""),
        };

        let mut cmd = Command::new(make_cmd);
        cmd.current_dir(&openblas_dir)
            .arg(format!("TARGET={}", openblas_target))
            .arg("NO_LAPACK=1") // Don't build LAPACK (we only need BLAS)
            .arg("NO_FORTRAN=1") // Don't require Fortran compiler
            .arg("ONLY_CBLAS=1") // Only build CBLAS interface
            .arg("NO_SHARED=1") // Only build static library
            .arg("USE_THREAD=1") // Enable threading
            .arg("NUM_THREADS=64") // Max threads
            .arg("libs") // Build only libraries target
            .arg(format!("-j{}", num_cpus::get()));

        if !cc_env.is_empty() {
            cmd.arg(cc_env);
        }

        // Cross-compilation support
        if let Ok(cross_compile) = std::env::var("CROSS_COMPILE") {
            cmd.arg(format!("CROSS={}", cross_compile));
        }

        let status = cmd.status().expect("Failed to build OpenBLAS");

        if !status.success() {
            panic!("OpenBLAS build failed for target: {} on {}", openblas_target, target_os);
        }

        println!(
            "cargo:warning=OpenBLAS built successfully for {} ({})",
            openblas_target, target_os
        );
    }

    let mut build = cc::Build::new();

    // Add source files
    build
        .file("kernels/ops_binary.c")
        .file("kernels/ops_concat_split.c")
        .file("kernels/ops_conv.c")
        .file("kernels/ops_indexing.c")
        .file("kernels/ops_matrix.c")
        .file("kernels/ops_reduce.c")
        .file("kernels/ops_unary.c")
        .file("kernels/ops_windowing.c")
        .include("kernels");

    // Standard flags
    build
        .flag_if_supported("-std=c11")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra");

    // Optimization flags
    build.flag_if_supported("-O3").flag_if_supported("-funroll-loops");

    // Safe fast math flags (preserve NaN/Inf behavior)
    build
        .flag_if_supported("-fno-math-errno")
        .flag_if_supported("-fno-trapping-math");

    // Vectorization flags
    build
        .flag_if_supported("-fvectorize")
        .flag_if_supported("-fslp-vectorize")
        .flag_if_supported("-ftree-vectorize");

    // Platform-specific optimizations (unless disabled)
    let disable_native = std::env::var("HODU_DISABLE_NATIVE").is_ok();
    if !disable_native {
        build
            .flag_if_supported("-march=native")
            .flag_if_supported("-mtune=native")
            .flag_if_supported("-mcpu=native"); // ARM equivalent
    }

    // SIMD auto-detection define (unless disabled)
    let disable_simd = std::env::var("HODU_DISABLE_SIMD").is_ok();
    if !disable_simd {
        build.define("ENABLE_SIMD_AUTO", None);
    }

    // Enable BLAS usage
    build.define("USE_BLAS", None);

    // Add OpenBLAS include path
    if openblas_dir.exists() {
        build.include(&openblas_dir);
    }

    // Embedded-friendly flags
    build
        .flag_if_supported("-fno-exceptions")
        .flag_if_supported("-fno-rtti");

    build.compile("hodu_cpu_kernels");

    // Link OpenBLAS - auto-detect library file regardless of version
    if openblas_dir.exists() {
        // Find any libopenblas*.a file (may have version suffix like libopenblas_armv8p-r0.3.30.dev.a)
        // Exclude symlinks - we want the actual file
        let openblas_lib_path = std::fs::read_dir(&openblas_dir).ok().and_then(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|entry| {
                    // Skip symlinks, only use real files
                    entry.file_type().map(|ft| !ft.is_symlink()).unwrap_or(false)
                })
                .find(|entry| {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    name_str.starts_with("libopenblas") && name_str.ends_with(".a")
                })
                .map(|entry| entry.path())
        });

        if let Some(lib_path) = openblas_lib_path {
            // Extract library name without "lib" prefix and ".a" suffix
            let lib_name = lib_path.file_stem().unwrap().to_string_lossy();
            let lib_name = lib_name.strip_prefix("lib").unwrap_or(&lib_name);

            // Use absolute path for link search to ensure linker can find the library
            let abs_dir = openblas_dir.canonicalize().expect("Failed to get absolute path");
            println!("cargo:rustc-link-search=native={}", abs_dir.display());
            println!("cargo:rustc-link-lib=static={}", lib_name);

            // Platform-specific system libraries
            let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
            let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

            match target_os.as_str() {
                "macos" => {
                    // macOS: link Accelerate framework (provides BLAS/LAPACK)
                    println!("cargo:rustc-link-lib=framework=Accelerate");
                },
                "linux" => {
                    // Linux: link pthread, and optionally gfortran
                    println!("cargo:rustc-link-lib=pthread");
                    if target_env != "musl" {
                        // glibc environments typically have gfortran
                        println!("cargo:rustc-link-lib=gfortran");
                    }
                },
                "windows" => {
                    // Windows: OpenBLAS typically needs these
                    println!("cargo:rustc-link-lib=gfortran");
                    println!("cargo:rustc-link-lib=quadmath");
                    println!("cargo:rustc-link-lib=gcc");
                },
                _ => {
                    // Other Unix-like systems
                    println!("cargo:rustc-link-lib=pthread");
                },
            }
        } else {
            println!("cargo:warning=OpenBLAS directory exists but no library file found. Skipping BLAS linking.");
        }
    }

    println!("cargo:rerun-if-changed=libs/OpenBLAS");

    // Tell cargo to rerun if C files change
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=kernels/atomic.h");
    println!("cargo:rerun-if-changed=kernels/constants.h");
    println!("cargo:rerun-if-changed=kernels/math_utils.h");
    println!("cargo:rerun-if-changed=kernels/types.h");
    println!("cargo:rerun-if-changed=kernels/utils.h");
    println!("cargo:rerun-if-changed=kernels/ops_binary.h");
    println!("cargo:rerun-if-changed=kernels/ops_binary.c");
    println!("cargo:rerun-if-changed=kernels/ops_concat_split.h");
    println!("cargo:rerun-if-changed=kernels/ops_concat_split.c");
    println!("cargo:rerun-if-changed=kernels/ops_conv.h");
    println!("cargo:rerun-if-changed=kernels/ops_conv.c");
    println!("cargo:rerun-if-changed=kernels/ops_indexing.h");
    println!("cargo:rerun-if-changed=kernels/ops_indexing.c");
    println!("cargo:rerun-if-changed=kernels/ops_matrix.h");
    println!("cargo:rerun-if-changed=kernels/ops_matrix.c");
    println!("cargo:rerun-if-changed=kernels/ops_reduce.h");
    println!("cargo:rerun-if-changed=kernels/ops_reduce.c");
    println!("cargo:rerun-if-changed=kernels/ops_unary.h");
    println!("cargo:rerun-if-changed=kernels/ops_unary.c");
    println!("cargo:rerun-if-changed=kernels/ops_windowing.h");
    println!("cargo:rerun-if-changed=kernels/ops_windowing.c");
}
