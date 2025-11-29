//! Run command - execute a .hdss model

use crate::output::{self, OutputFormat};
use hodu_core::{
    format::{hdss, hdt, json},
    script::Script,
    tensor::Tensor,
};
use hodu_plugin::{Device, HoduError, HoduResult, InterpRuntime, PluginManager, RuntimePlugin, TensorData};
use std::collections::HashMap;
use std::path::PathBuf;

fn parse_device(s: &str) -> HoduResult<Device> {
    match s.to_lowercase().as_str() {
        "cpu" => Ok(Device::CPU),
        #[cfg(feature = "cuda")]
        s if s.starts_with("cuda:") => {
            let id: usize = s[5..]
                .parse()
                .map_err(|_| HoduError::InvalidArgument("Invalid CUDA device ID".into()))?;
            Ok(Device::CUDA(id))
        },
        "metal" => Ok(Device::Metal),
        _ => Err(HoduError::InvalidArgument(format!(
            "Unknown device: {}. Use 'cpu', 'metal'{}",
            s,
            if cfg!(feature = "cuda") { ", 'cuda:N'" } else { "" },
        ))),
    }
}

fn load_tensor(path: &PathBuf) -> HoduResult<Tensor> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "hdt" => hdt::load(path),
        "json" => json::load(path),
        _ => Err(HoduError::InvalidArgument(format!(
            "Unsupported tensor format: {}. Use .hdt or .json",
            ext
        ))),
    }
}

fn parse_input(s: &str) -> HoduResult<(String, PathBuf)> {
    let parts: Vec<&str> = s.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(HoduError::InvalidArgument(format!(
            "Invalid input format: '{}'. Use name=path.hdt or name=path.json",
            s
        )));
    }
    Ok((parts[0].to_string(), PathBuf::from(parts[1])))
}

pub fn execute(
    path: PathBuf,
    device_str: &str,
    input: Vec<String>,
    inputs: Vec<String>,
    output_format: OutputFormat,
    output_dir: Option<PathBuf>,
    compiler_plugin: Option<PathBuf>,
    runtime_plugin: Option<PathBuf>,
) -> HoduResult<()> {
    let device = parse_device(device_str)?;
    let snapshot = hdss::load(&path)?;

    // Merge --input and --inputs
    let all_inputs: Vec<&str> = input.iter().chain(inputs.iter()).map(|s| s.as_str()).collect();

    // Parse and load input tensors
    let input_map: HashMap<String, Tensor> = if !all_inputs.is_empty() {
        all_inputs
            .iter()
            .map(|s| {
                let (name, path) = parse_input(s)?;
                let tensor = load_tensor(&path)?;
                Ok((name, tensor))
            })
            .collect::<HoduResult<HashMap<_, _>>>()?
    } else if !snapshot.inputs.is_empty() {
        return Err(HoduError::InvalidArgument(format!(
            "Missing inputs. Required: {}",
            snapshot
                .inputs
                .iter()
                .map(|i| format!("{} ({:?} {:?})", i.name, i.dtype, i.shape))
                .collect::<Vec<_>>()
                .join(", ")
        )));
    } else {
        HashMap::new()
    };

    // Validate all required inputs are provided
    for input in &snapshot.inputs {
        if !input_map.contains_key(&input.name) {
            return Err(HoduError::InvalidArgument(format!(
                "Missing input: '{}'. Required: {}",
                input.name,
                snapshot
                    .inputs
                    .iter()
                    .map(|i| i.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
    }

    // Convert input tensors to TensorData
    let input_data: Vec<(String, TensorData)> = snapshot
        .inputs
        .iter()
        .map(|input| {
            let tensor = input_map.get(&input.name).unwrap();
            let data = tensor.to_bytes()?;
            let shape = tensor.shape().dims().to_vec();
            let dtype = tensor.dtype();
            Ok((input.name.clone(), TensorData::new(data, shape, dtype)))
        })
        .collect::<HoduResult<Vec<_>>>()?;

    let input_bindings: Vec<(&str, TensorData)> = input_data
        .iter()
        .map(|(name, data)| (name.as_str(), data.clone()))
        .collect();

    // Execute based on device
    let outputs: HashMap<String, TensorData> = match device {
        Device::CPU if compiler_plugin.is_some() => {
            // Use compiler + runtime plugins for CPU when specified
            let compiler_path = compiler_plugin.unwrap();

            // Load compiler plugin
            let mut manager = PluginManager::with_default_dir()?;
            manager.load_compiler(&compiler_path)?;

            // Find compiler
            let compiler = manager
                .compilers()
                .find(|c| c.supports_device(device))
                .ok_or_else(|| HoduError::BackendError("No CPU compiler found".into()))?;

            // Compile
            let script = Script::new(snapshot.clone());
            let artifact = compiler.compile(&script, device)?;

            // Use runtime plugin if provided, otherwise use interpreter
            if let Some(runtime_path) = runtime_plugin {
                manager.load_runtime(&runtime_path)?;
                // Prefer runtime that supports SharedLib format (the dynamically loaded one)
                let runtime = manager
                    .runtimes()
                    .find(|r| {
                        r.supports_device(device)
                            && r.loadable_formats(device)
                                .contains(&hodu_plugin::OutputFormat::SharedLib)
                    })
                    .ok_or_else(|| HoduError::BackendError("No CPU runtime supporting SharedLib found".into()))?;
                let module = runtime.load(&artifact, device)?;
                module.execute(&input_bindings)?
            } else {
                // Use interpreter runtime with compiled artifact
                let runtime = InterpRuntime::new();
                let module = runtime.load(&artifact, Device::CPU)?;
                module.execute(&input_bindings)?
            }
        },
        Device::CPU => {
            // Use interpreter runtime for CPU (no plugins)
            let runtime = InterpRuntime::new();
            let snapshot_data = snapshot.serialize()?;
            let artifact =
                hodu_plugin::CompiledArtifact::new(hodu_plugin::OutputFormat::HoduSnapshot, Device::CPU, snapshot_data);
            let module = runtime.load(&artifact, Device::CPU)?;
            module.execute(&input_bindings)?
        },
        Device::Metal => {
            // Use compiler + runtime plugins for Metal
            let compiler_path = compiler_plugin
                .ok_or_else(|| HoduError::InvalidArgument("Metal device requires --compiler-plugin path".into()))?;
            let runtime_path = runtime_plugin
                .ok_or_else(|| HoduError::InvalidArgument("Metal device requires --runtime-plugin path".into()))?;

            // Load plugins
            let mut manager = PluginManager::with_default_dir()?;
            manager.load_compiler(&compiler_path)?;
            manager.load_runtime(&runtime_path)?;

            // Find compiler and runtime
            let compiler = manager
                .compilers()
                .find(|c| c.supports_device(device))
                .ok_or_else(|| HoduError::BackendError("No Metal compiler found".into()))?;
            let runtime = manager
                .runtimes()
                .find(|r| r.supports_device(device))
                .ok_or_else(|| HoduError::BackendError("No Metal runtime found".into()))?;

            // Compile
            let script = Script::new(snapshot.clone());
            let artifact = compiler.compile(&script, device)?;

            // Load and execute
            let module = runtime.load(&artifact, device)?;
            module.execute(&input_bindings)?
        },
        #[cfg(feature = "cuda")]
        Device::CUDA(_) => {
            return Err(HoduError::UnsupportedDevice(device));
        },
        #[allow(unreachable_patterns)]
        _ => {
            return Err(HoduError::UnsupportedDevice(device));
        },
    };

    // Output in requested format
    let target_order: Vec<String> = snapshot.targets.iter().map(|t| t.name.clone()).collect();
    output::print_outputs(&outputs, &target_order, output_format, output_dir.as_deref())
}
