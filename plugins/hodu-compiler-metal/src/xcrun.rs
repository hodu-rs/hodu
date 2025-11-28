//! xcrun wrapper for Metal compilation
//!
//! Uses macOS xcrun to compile Metal shaders:
//! - `xcrun metal` : .metal → .air
//! - `xcrun metallib` : .air → .metallib

use hodu_plugin::{HoduError, HoduResult};
use std::path::Path;
use std::process::Command;

/// Compile Metal source to AIR (Apple Intermediate Representation)
pub fn compile_metal_to_air(metal_path: &Path, air_path: &Path) -> HoduResult<()> {
    let output = Command::new("xcrun")
        .args([
            "metal",
            "-c",
            metal_path.to_str().unwrap_or(""),
            "-o",
            air_path.to_str().unwrap_or(""),
        ])
        .output()
        .map_err(|e| HoduError::BackendError(format!("xcrun metal failed: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(HoduError::BackendError(format!(
            "xcrun metal compilation failed:\n{}",
            stderr
        )));
    }

    Ok(())
}

/// Create metallib from AIR files
pub fn create_metallib(air_paths: &[&Path], metallib_path: &Path) -> HoduResult<()> {
    let mut args = vec!["metallib", "-o", metallib_path.to_str().unwrap_or("")];
    for air_path in air_paths {
        args.push(air_path.to_str().unwrap_or(""));
    }

    let output = Command::new("xcrun")
        .args(&args)
        .output()
        .map_err(|e| HoduError::BackendError(format!("xcrun metallib failed: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(HoduError::BackendError(format!(
            "xcrun metallib creation failed:\n{}",
            stderr
        )));
    }

    Ok(())
}

/// Compile Metal source directly to metallib
pub fn compile_metal_to_metallib(metal_path: &Path, metallib_path: &Path) -> HoduResult<()> {
    // Create temp AIR file
    let air_path = metal_path.with_extension("air");

    // Compile .metal → .air
    compile_metal_to_air(metal_path, &air_path)?;

    // Create .metallib from .air
    create_metallib(&[&air_path], metallib_path)?;

    // Clean up temp AIR file
    let _ = std::fs::remove_file(&air_path);

    Ok(())
}

/// Compile multiple Metal sources to a single metallib
#[allow(dead_code)]
pub fn compile_metals_to_metallib(metal_paths: &[&Path], metallib_path: &Path) -> HoduResult<()> {
    let mut air_paths = Vec::new();

    // Compile each .metal → .air
    for metal_path in metal_paths {
        let air_path = metal_path.with_extension("air");
        compile_metal_to_air(metal_path, &air_path)?;
        air_paths.push(air_path);
    }

    // Create .metallib from all .air files
    let air_refs: Vec<&Path> = air_paths.iter().map(|p| p.as_path()).collect();
    create_metallib(&air_refs, metallib_path)?;

    // Clean up temp AIR files
    for air_path in air_paths {
        let _ = std::fs::remove_file(&air_path);
    }

    Ok(())
}

/// Check if xcrun metal is available
pub fn is_available() -> bool {
    Command::new("xcrun")
        .args(["--find", "metal"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
