use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let kernels_dir = Path::new("../../crates/hodu_cpu_kernels/kernels");

    // Build CPU kernels
    let mut build = cc::Build::new();

    build
        .file(kernels_dir.join("ops_binary.c"))
        .file(kernels_dir.join("ops_cast.c"))
        .file(kernels_dir.join("ops_concat_split.c"))
        .file(kernels_dir.join("ops_conv.c"))
        .file(kernels_dir.join("ops_indexing.c"))
        .file(kernels_dir.join("ops_matrix.c"))
        .file(kernels_dir.join("ops_memory.c"))
        .file(kernels_dir.join("ops_reduce.c"))
        .file(kernels_dir.join("ops_unary.c"))
        .file(kernels_dir.join("ops_windowing.c"))
        .file(kernels_dir.join("storage.c"))
        .include(kernels_dir);

    // Standard flags
    build
        .flag_if_supported("-std=c11")
        .flag_if_supported("-Wall")
        .flag_if_supported("-O3")
        .flag_if_supported("-funroll-loops")
        .flag_if_supported("-fno-math-errno")
        .flag_if_supported("-fno-trapping-math")
        .flag_if_supported("-fvectorize")
        .flag_if_supported("-fslp-vectorize")
        .flag_if_supported("-ftree-vectorize");

    // Platform-specific optimizations
    if env::var("HODU_DISABLE_NATIVE").is_err() {
        build
            .flag_if_supported("-march=native")
            .flag_if_supported("-mtune=native")
            .flag_if_supported("-mcpu=native");
    }

    // SIMD auto-detection
    if env::var("HODU_DISABLE_SIMD").is_err() {
        build.define("ENABLE_SIMD_AUTO", None);
    }

    // Thread parallelization
    if env::var("HODU_DISABLE_THREADS").is_err() {
        build.define("ENABLE_THREADS", None);
    }

    // Embedded-friendly flags
    build
        .flag_if_supported("-fno-exceptions")
        .flag_if_supported("-fno-rtti");

    // Compile with a custom name to avoid conflicts
    build.compile("hodu_cpu_kernels_bundled");

    // macOS Accelerate framework for BLAS
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        // Get SDK path for Accelerate framework headers
        let sdk_path = std::process::Command::new("xcrun")
            .args(["--show-sdk-path"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

        // Add BLAS implementations for macOS
        let mut blas_build = cc::Build::new();
        blas_build
            .file(kernels_dir.join("ops_conv_blas_aarch64_apple_darwin.c"))
            .file(kernels_dir.join("ops_matrix_blas_aarch64_apple_darwin.c"))
            .file(kernels_dir.join("ops_unary_blas_aarch64_apple_darwin.c"))
            .include(kernels_dir)
            .flag_if_supported("-std=c11")
            .flag_if_supported("-O3")
            .define("USE_BLAS", None)
            .define("ACCELERATE_NEW_LAPACK", None);

        // Include Accelerate framework headers
        if let Some(sdk) = sdk_path {
            let accelerate_include = format!(
                "{}/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers",
                sdk
            );
            blas_build.include(&accelerate_include);
        }

        blas_build.compile("hodu_cpu_kernels_blas");

        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Link pthread
    if env::var("HODU_DISABLE_THREADS").is_err() {
        match target_os.as_str() {
            "linux" | "macos" => println!("cargo:rustc-link-lib=pthread"),
            "windows" => {
                let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
                if target_env == "gnu" {
                    println!("cargo:rustc-link-lib=pthread");
                }
            }
            _ => {}
        }
    }

    // Read the compiled library and write it as a Rust byte array
    let lib_path = Path::new(&out_dir).join("libhodu_cpu_kernels_bundled.a");

    // Also include BLAS library on macOS
    let blas_lib_path = Path::new(&out_dir).join("libhodu_cpu_kernels_blas.a");

    // Generate Rust code to embed the libraries
    let dest_path = Path::new(&out_dir).join("bundled_kernels.rs");

    let mut code = String::new();
    code.push_str("// Auto-generated bundled CPU kernels\n\n");

    // Embed main kernels library
    code.push_str(&format!(
        "pub static KERNELS_LIB: &[u8] = include_bytes!(\"{}\");\n",
        lib_path.to_string_lossy().replace('\\', "/")
    ));

    // Embed BLAS library if it exists (macOS)
    if blas_lib_path.exists() {
        code.push_str(&format!(
            "pub static BLAS_LIB: Option<&[u8]> = Some(include_bytes!(\"{}\"));\n",
            blas_lib_path.to_string_lossy().replace('\\', "/")
        ));
    } else {
        code.push_str("pub static BLAS_LIB: Option<&[u8]> = None;\n");
    }

    fs::write(&dest_path, code).unwrap();

    // Rerun triggers
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", kernels_dir.display());
}
