//! Run command - execute a .hdss model

use hodu_core::tensor::Tensor;
use hodu_format::{hdss, hdt, json};
use hodu_plugin::{Device, HoduError, HoduResult, InterpRuntime};
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
        #[cfg(feature = "metal")]
        "metal" => Ok(Device::Metal),
        _ => Err(HoduError::InvalidArgument(format!(
            "Unknown device: {}. Use 'cpu'{}{}",
            s,
            if cfg!(feature = "cuda") { ", 'cuda:N'" } else { "" },
            if cfg!(feature = "metal") { ", 'metal'" } else { "" },
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

pub fn execute(path: PathBuf, device_str: &str, input: Vec<String>, inputs: Vec<String>) -> HoduResult<()> {
    let _device = parse_device(device_str)?;
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

    let runtime = InterpRuntime::new();
    let input_bindings: Vec<(&str, &Tensor)> = snapshot
        .inputs
        .iter()
        .map(|input| {
            let tensor = input_map.get(&input.name).unwrap();
            (input.name.as_str(), tensor)
        })
        .collect();
    let outputs = runtime.execute_snapshot(&snapshot, &input_bindings)?;

    // Print outputs in target order
    for target in &snapshot.targets {
        if let Some(tensor) = outputs.get(&target.name) {
            println!("{}", tensor);
        }
    }

    Ok(())
}
