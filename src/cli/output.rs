//! Output formatting for CLI

use clap::ValueEnum;
use hodu_core::{tensor::Tensor, types::DType};
use hodu_plugin::{HoduError, HoduResult, TensorData};
use std::collections::HashMap;
use std::path::Path;

/// Output format for run command
#[derive(Clone, Copy, Default, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable format: name: dtype[shape] = data
    #[default]
    Pretty,
    /// JSON format for scripting
    Json,
    /// Save to .hdt files (requires --output-dir)
    Hdt,
}

/// Format and print outputs
pub fn print_outputs(
    outputs: &HashMap<String, TensorData>,
    target_order: &[String],
    format: OutputFormat,
    output_dir: Option<&Path>,
) -> HoduResult<()> {
    match format {
        OutputFormat::Pretty => print_pretty(outputs, target_order),
        OutputFormat::Json => print_json(outputs, target_order),
        OutputFormat::Hdt => save_hdt(outputs, output_dir),
    }
}

/// Pretty print: name: dtype[shape] = data
fn print_pretty(outputs: &HashMap<String, TensorData>, target_order: &[String]) -> HoduResult<()> {
    for name in target_order {
        if let Some(tensor_data) = outputs.get(name) {
            let tensor = tensor_from_data(tensor_data)?;
            let shape_str = format_shape(&tensor_data.shape);
            let dtype_str = format_dtype(tensor_data.dtype);

            // Format tensor data as single line if small, truncate if large
            let data_str = format_tensor_data(&tensor, 100);

            println!("{}: {}[{}] = {}", name, dtype_str, shape_str, data_str);
        }
    }
    Ok(())
}

/// JSON output for scripting
fn print_json(outputs: &HashMap<String, TensorData>, target_order: &[String]) -> HoduResult<()> {
    use std::fmt::Write;

    let mut json = String::from("{");
    let mut first = true;

    for name in target_order {
        if let Some(tensor_data) = outputs.get(name) {
            if !first {
                json.push_str(", ");
            }
            first = false;

            let tensor = tensor_from_data(tensor_data)?;
            let shape_json = format!("{:?}", tensor_data.shape);
            let dtype_str = format_dtype(tensor_data.dtype);
            let data_json = tensor_to_json_array(&tensor);

            write!(
                json,
                "\"{}\": {{\"dtype\": \"{}\", \"shape\": {}, \"data\": {}}}",
                name, dtype_str, shape_json, data_json
            )
            .unwrap();
        }
    }

    json.push('}');
    println!("{}", json);
    Ok(())
}

/// Save outputs as .hdt files
fn save_hdt(outputs: &HashMap<String, TensorData>, output_dir: Option<&Path>) -> HoduResult<()> {
    let dir = output_dir.ok_or_else(|| HoduError::InvalidArgument("--output-dir required for hdt format".into()))?;

    std::fs::create_dir_all(dir).map_err(|e| HoduError::IoError(format!("Failed to create output dir: {}", e)))?;

    for (name, tensor_data) in outputs {
        let tensor = tensor_from_data(tensor_data)?;
        let path = dir.join(format!("{}.hdt", name));
        hodu_core::format::hdt::save(&tensor, &path)?;
        println!("Saved: {}", path.display());
    }

    Ok(())
}

fn tensor_from_data(data: &TensorData) -> HoduResult<Tensor> {
    Tensor::from_bytes(&data.data, &data.shape, data.dtype, hodu_plugin::Device::CPU)
}

fn format_shape(shape: &[usize]) -> String {
    shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
}

fn format_dtype(dtype: DType) -> &'static str {
    match dtype {
        DType::BOOL => "bool",
        DType::F8E4M3 => "f8e4m3",
        DType::F8E5M2 => "f8e5m2",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::U8 => "u8",
        DType::U16 => "u16",
        DType::U32 => "u32",
        DType::U64 => "u64",
        DType::I8 => "i8",
        DType::I16 => "i16",
        DType::I32 => "i32",
        DType::I64 => "i64",
    }
}

/// Format tensor data, truncating if too large
fn format_tensor_data(tensor: &Tensor, max_elements: usize) -> String {
    let total = tensor.shape().size();
    let display = tensor.to_string();

    if total > max_elements {
        // Find a good truncation point
        let truncated: String = display.chars().take(200).collect();
        format!("{} ... (truncated, {} elements)", truncated, total)
    } else {
        // Single line format
        display.replace('\n', "").replace("  ", " ")
    }
}

/// Convert tensor to JSON array format
fn tensor_to_json_array(tensor: &Tensor) -> String {
    // Use the tensor's display and convert to JSON-like format
    let s = tensor.to_string();
    s.replace('\n', "")
        .replace("  ", " ")
        .replace(" ,", ",")
        .replace(", ", ",")
        .replace("[ ", "[")
        .replace(" ]", "]")
}
