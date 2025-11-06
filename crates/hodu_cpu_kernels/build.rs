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

    // Embedded-friendly flags
    build
        .flag_if_supported("-fno-exceptions")
        .flag_if_supported("-fno-rtti");

    build.compile("hodu_cpu_kernels");

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
