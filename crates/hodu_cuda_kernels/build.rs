use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir.join("kernels");

    // Tell cargo to rerun if any .cu or .cuh files change
    println!("cargo:rerun-if-changed=kernels/");

    // List of kernel files to compile
    let kernel_files = vec![
        "ops_binary.cu",
        "ops_cast.cu",
        "ops_concat_split.cu",
        "ops_conv.cu",
        "ops_indexing.cu",
        "ops_matrix.cu",
        "ops_memory.cu",
        "ops_padding.cu",
        "ops_reduce.cu",
        "ops_scan.cu",
        "ops_shape_memory.cu",
        "ops_unary.cu",
        "ops_windowing.cu",
        "storage.cu",
    ];

    // Compile each kernel to PTX
    for kernel_file in &kernel_files {
        let input_path = kernels_dir.join(kernel_file);
        let output_name = kernel_file.replace(".cu", ".ptx");
        let output_path = out_dir.join(&output_name);

        let err_msg = format!("Failed to run nvcc for {}. Make sure nvcc is in PATH.", kernel_file);

        let status = Command::new("nvcc")
            .arg("--ptx")
            .arg(&input_path)
            .arg("-o")
            .arg(&output_path)
            .arg("--fmad=true")
            .arg("--expt-relaxed-constexpr")
            .arg(format!("-I{}", kernels_dir.display()))
            .status()
            .expect(&err_msg);

        if !status.success() {
            panic!("nvcc compilation failed for {}", kernel_file);
        }
    }

    // Generate source.rs with PTX includes
    generate_source_rs(&out_dir, &kernel_files);
}

fn generate_source_rs(out_dir: &Path, kernel_files: &[&str]) {
    let source_rs_path = out_dir.join("generated_source.rs");

    let mut content = String::new();

    // Generate PTX includes
    for kernel_file in kernel_files {
        let ptx_name = kernel_file.replace(".cu", ".ptx");
        let var_name = kernel_file.replace(".cu", "").to_uppercase();

        content.push_str(&format!(
            "const {}_PTX: &str = include_str!(concat!(env!(\"OUT_DIR\"), \"/{}\"));\n",
            var_name, ptx_name
        ));
    }

    content.push('\n');

    // Generate getter functions
    for kernel_file in kernel_files {
        let var_name = kernel_file.replace(".cu", "").to_uppercase();
        let fn_name = format!("get_{}", kernel_file.replace(".cu", ""));

        content.push_str(&format!(
            "pub fn {}() -> &'static str {{\n    {}_PTX\n}}\n\n",
            fn_name, var_name
        ));
    }

    fs::write(source_rs_path, content).expect("Failed to write generated_source.rs");
}
