//! Plugin creation/scaffolding logic

use std::path::PathBuf;

pub fn create_plugin(name: &str, plugin_type: &str, output: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    use hodu_plugin_sdk::{
        cargo_toml_template, main_rs_backend_template, main_rs_model_format_template, main_rs_tensor_format_template,
        manifest_json_backend_template, manifest_json_model_format_template, manifest_json_tensor_format_template,
    };

    let plugin_type = plugin_type.to_lowercase();
    let valid_types = ["backend", "model_format", "tensor_format"];
    if !valid_types.contains(&plugin_type.as_str()) {
        return Err(format!(
            "Invalid plugin type: '{}'. Use 'backend', 'model_format', or 'tensor_format'.",
            plugin_type
        )
        .into());
    }

    let output_dir = match output {
        Some(dir) => dir,
        None => std::env::current_dir().map_err(|e| format!("Failed to get current directory: {}", e))?,
    };
    let project_dir = output_dir.join(name);

    if project_dir.exists() {
        return Err(format!("Directory already exists: {}", project_dir.display()).into());
    }

    println!("Creating {} plugin: {}", plugin_type, name);

    std::fs::create_dir_all(&project_dir)?;
    std::fs::create_dir_all(project_dir.join("src"))?;

    // Cargo.toml
    let cargo_toml = cargo_toml_template(name);
    std::fs::write(project_dir.join("Cargo.toml"), cargo_toml)?;

    // manifest.json
    let manifest = match plugin_type.as_str() {
        "backend" => manifest_json_backend_template(name),
        "model_format" => manifest_json_model_format_template(name),
        "tensor_format" => manifest_json_tensor_format_template(name),
        _ => unreachable!(),
    };
    std::fs::write(project_dir.join("manifest.json"), manifest)?;

    // main.rs
    let main_rs = match plugin_type.as_str() {
        "backend" => main_rs_backend_template(name),
        "model_format" => main_rs_model_format_template(name),
        "tensor_format" => main_rs_tensor_format_template(name),
        _ => unreachable!(),
    };
    std::fs::write(project_dir.join("src").join("main.rs"), main_rs)?;

    println!("Created plugin project at: {}", project_dir.display());
    println!();
    println!("Next steps:");
    println!("  1. cd {}", name);
    println!("  2. Edit manifest.json with your plugin details");
    println!("  3. Implement the plugin in src/main.rs");
    println!("  4. Install with: hodu plugin install --path .");

    Ok(())
}
