mod build_openblas;

fn main() {
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

    // Thread parallelization define (unless disabled, std feature only)
    #[cfg(feature = "std")]
    {
        let disable_threads = std::env::var("HODU_DISABLE_THREADS").is_ok();
        if !disable_threads {
            build.define("ENABLE_THREADS", None);
        }
    }

    // Detect and configure OpenBLAS (automatically skips for no_std targets)
    let openblas_config = build_openblas::OpenBlasConfig::detect();
    openblas_config.print_status();
    openblas_config.apply_to_build(&mut build);

    // Embedded-friendly flags
    build
        .flag_if_supported("-fno-exceptions")
        .flag_if_supported("-fno-rtti");

    build.compile("hodu_cpu_kernels");

    // Link pthread for multi-threaded kernel execution (std feature only, unless disabled)
    #[cfg(feature = "std")]
    {
        let disable_threads = std::env::var("HODU_DISABLE_THREADS").is_ok();
        if !disable_threads {
            let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
            let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

            match target_os.as_str() {
                "linux" | "macos" => {
                    println!("cargo:rustc-link-lib=pthread");
                },
                "windows" => {
                    if target_env == "gnu" {
                        // MinGW: use winpthread
                        println!("cargo:rustc-link-lib=pthread");
                    }
                    // MSVC: native Windows threads (no extra linking needed)
                },
                _ => {},
            }
        }
    }

    // Setup OpenBLAS linking
    openblas_config.setup_linking();

    // Tell cargo to rerun if C files change
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=kernels/atomic.h");
    println!("cargo:rerun-if-changed=kernels/constants.h");
    println!("cargo:rerun-if-changed=kernels/math_utils.h");
    println!("cargo:rerun-if-changed=kernels/simd_utils.h");
    println!("cargo:rerun-if-changed=kernels/thread_utils.h");
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
