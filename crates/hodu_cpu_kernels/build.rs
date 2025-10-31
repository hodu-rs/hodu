fn main() {
    // Compile C kernels
    cc::Build::new()
        .file("kernels/ops_binary.c")
        .file("kernels/ops_concat_split.c")
        .file("kernels/ops_conv.c")
        .file("kernels/ops_indexing.c")
        .file("kernels/ops_matrix.c")
        .file("kernels/ops_reduce.c")
        .file("kernels/ops_unary.c")
        .file("kernels/ops_windowing.c")
        .include("kernels")
        .flag_if_supported("-std=c11")
        .flag_if_supported("-O3")
        // Embedded-friendly flags
        .flag_if_supported("-fno-exceptions")
        .flag_if_supported("-fno-rtti")
        // Warning flags
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .compile("hodu_cpu_kernels");

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
