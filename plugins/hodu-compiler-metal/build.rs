use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let kernels_dir = Path::new("../../crates/hodu_metal_kernels/kernels");

    // Order matters: headers first
    let files = [
        "headers/constants.metal",
        "headers/utils.metal",
        "headers/atomic.metal",
        "storage.metal",
        "ops_binary.metal",
        "ops_unary.metal",
        "ops_reduce.metal",
        "ops_matrix.metal",
        "ops_cast.metal",
        "ops_memory.metal",
        "ops_indexing.metal",
        "ops_concat_split.metal",
        "ops_conv.metal",
        "ops_windowing.metal",
    ];

    // Lines that should only appear once (from headers)
    let duplicate_patterns = [
        "template <typename T> T maximum(",
        "template <typename T> T minimum(",
    ];

    let mut combined = String::from("// Bundled Hodu Metal Kernels\n\n");
    combined.push_str("#include <metal_stdlib>\n");
    combined.push_str("using namespace metal;\n\n");

    let mut seen_patterns: std::collections::HashSet<&str> = std::collections::HashSet::new();

    for file in &files {
        let path = kernels_dir.join(file);
        if path.exists() {
            let content = fs::read_to_string(&path).unwrap_or_default();
            let is_header = file.starts_with("headers/");

            let filtered: String = content
                .lines()
                .filter(|line| {
                    // Skip common includes/pragmas
                    if line.starts_with("#include <metal_stdlib>")
                        || line.starts_with("using namespace metal;")
                        || line.starts_with("#include \"./headers/")
                        || line.starts_with("#pragma once")
                    {
                        return false;
                    }

                    // Handle duplicate helper functions
                    for pattern in &duplicate_patterns {
                        if line.contains(pattern) {
                            if is_header {
                                // First occurrence in headers - keep it
                                seen_patterns.insert(pattern);
                                return true;
                            } else if seen_patterns.contains(pattern) {
                                // Already defined in headers - skip
                                return false;
                            }
                        }
                    }

                    true
                })
                .collect::<Vec<_>>()
                .join("\n");
            combined.push_str(&format!("// === {} ===\n", file));
            combined.push_str(&filtered);
            combined.push_str("\n\n");
        }
    }

    let dest_path = Path::new(&out_dir).join("bundled_kernels.metal");
    fs::write(&dest_path, combined).unwrap();

    println!("cargo:rerun-if-changed=build.rs");
    for file in &files {
        println!("cargo:rerun-if-changed={}", kernels_dir.join(file).display());
    }
}
