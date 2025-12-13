mod build_blas;
mod build_openblas;

fn main() {
    let mut build = cc::Build::new();

    // Source files
    build
        .file("kernels/ops_binary.c")
        .file("kernels/ops_cast.c")
        .file("kernels/ops_concat_split.c")
        .file("kernels/ops_conv.c")
        .file("kernels/ops_einsum.c")
        .file("kernels/ops_indexing.c")
        .file("kernels/ops_linalg.c")
        .file("kernels/ops_matrix.c")
        .file("kernels/ops_memory.c")
        .file("kernels/ops_padding.c")
        .file("kernels/ops_reduce.c")
        .file("kernels/ops_resize.c")
        .file("kernels/ops_scan.c")
        .file("kernels/ops_shape_memory.c")
        .file("kernels/ops_sort.c")
        .file("kernels/ops_unary.c")
        .file("kernels/ops_windowing.c")
        .file("kernels/storage.c")
        .include("kernels");

    // BLAS configuration (must be done before adding source files)
    configure_blas(&mut build);

    // BLAS-specific implementations
    add_blas_impl(&mut build);

    // Standard flags
    build
        .flag_if_supported("-std=c11")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .flag_if_supported("-O3")
        .flag_if_supported("-funroll-loops")
        .flag_if_supported("-fno-math-errno")
        .flag_if_supported("-fno-trapping-math")
        .flag_if_supported("-fvectorize")
        .flag_if_supported("-fslp-vectorize")
        .flag_if_supported("-ftree-vectorize");

    // Platform-specific optimizations
    if std::env::var("HODU_DISABLE_NATIVE").is_err() {
        build
            .flag_if_supported("-march=native")
            .flag_if_supported("-mtune=native")
            .flag_if_supported("-mcpu=native");
    }

    // SIMD auto-detection
    if std::env::var("HODU_DISABLE_SIMD").is_err() {
        build.define("ENABLE_SIMD_AUTO", None);
    }

    // Thread parallelization
    if std::env::var("HODU_DISABLE_THREADS").is_err() {
        build.define("ENABLE_THREADS", None);
    }

    // Embedded-friendly flags
    build
        .flag_if_supported("-fno-exceptions")
        .flag_if_supported("-fno-rtti");

    build.compile("hodu_cpu_kernels");

    // Link pthread
    link_pthread();

    // Link BLAS library
    link_blas();

    // Rerun triggers
    println!("cargo:rerun-if-changed=kernels/");
    for file in [
        "atomic.h",
        "constants.h",
        "math_utils.h",
        "simd_utils.h",
        "thread_utils.h",
        "types.h",
        "utils.h",
        "ops_binary.h",
        "ops_binary.c",
        "ops_cast.h",
        "ops_cast.c",
        "ops_concat_split.h",
        "ops_concat_split.c",
        "ops_conv.h",
        "ops_conv.c",
        "ops_einsum.h",
        "ops_einsum.c",
        "ops_indexing.h",
        "ops_indexing.c",
        "ops_linalg.h",
        "ops_linalg.c",
        "ops_matrix.h",
        "ops_matrix.c",
        "ops_matrix_openblas.c",
        "ops_matrix_blas_aarch64_apple_darwin.c",
        "ops_conv_openblas.c",
        "ops_conv_blas_aarch64_apple_darwin.c",
        "ops_unary_openblas.c",
        "ops_unary_blas_aarch64_apple_darwin.c",
        "ops_memory.h",
        "ops_memory.c",
        "ops_padding.h",
        "ops_padding.c",
        "ops_reduce.h",
        "ops_reduce.c",
        "ops_resize.h",
        "ops_resize.c",
        "ops_scan.h",
        "ops_scan.c",
        "ops_shape_memory.h",
        "ops_shape_memory.c",
        "ops_sort.h",
        "ops_sort.c",
        "ops_unary.h",
        "ops_unary.c",
        "ops_windowing.h",
        "ops_windowing.c",
        "storage.h",
        "storage.c",
    ] {
        println!("cargo:rerun-if-changed=kernels/{}", file);
    }
}

fn add_blas_impl(build: &mut cc::Build) {
    #[cfg(feature = "openblas")]
    {
        let openblas = build_openblas::OpenBlasConfig::detect();
        if openblas.available {
            build.file("kernels/ops_conv_openblas.c");
            build.file("kernels/ops_matrix_openblas.c");
            build.file("kernels/ops_unary_openblas.c");
            return;
        }
        openblas.warn_if_not_found();
    }

    #[cfg(not(feature = "openblas"))]
    {
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if target_os == "macos" {
            build.file("kernels/ops_conv_blas_aarch64_apple_darwin.c");
            build.file("kernels/ops_matrix_blas_aarch64_apple_darwin.c");
            build.file("kernels/ops_unary_blas_aarch64_apple_darwin.c");
        }
    }
}

fn configure_blas(build: &mut cc::Build) {
    #[cfg(feature = "openblas")]
    {
        let openblas = build_openblas::OpenBlasConfig::detect();
        if openblas.available {
            openblas.apply_to_build(build);
            return;
        }
        openblas.warn_if_not_found();
    }

    #[cfg(not(feature = "openblas"))]
    {
        build_blas::BlasConfig::detect().apply_to_build(build);
    }
}

fn link_blas() {
    #[cfg(feature = "openblas")]
    {
        let openblas = build_openblas::OpenBlasConfig::detect();
        if openblas.available {
            openblas.setup_linking();
        }
    }

    #[cfg(not(feature = "openblas"))]
    {
        build_blas::BlasConfig::detect().setup_linking();
    }
}

fn link_pthread() {
    if std::env::var("HODU_DISABLE_THREADS").is_ok() {
        return;
    }

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    match target_os.as_str() {
        "linux" | "macos" => println!("cargo:rustc-link-lib=pthread"),
        "windows" if target_env == "gnu" => println!("cargo:rustc-link-lib=pthread"),
        _ => {},
    }
}
