//! Run command - execute a .hdss model

use hodu_format::hdss;
use hodu_plugin::{Device, HoduError, HoduResult, InterpRuntime};
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

pub fn execute(path: PathBuf, device_str: &str) -> HoduResult<()> {
    let _device = parse_device(device_str)?;

    println!("Loading model from: {}", path.display());

    let snapshot = hdss::load(&path)?;

    println!("Model loaded successfully");
    println!("  Inputs: {}", snapshot.inputs.len());
    println!("  Nodes: {}", snapshot.nodes.len());
    println!("  Targets: {}", snapshot.targets.len());

    if !snapshot.inputs.is_empty() {
        println!("\nNote: This model requires {} input(s)", snapshot.inputs.len());
        println!("Input bindings:");
        for (i, input) in snapshot.inputs.iter().enumerate() {
            println!("  [{}] {:?} {:?}", i, input.dtype, input.shape);
        }
        println!("\nTo run with inputs, use the library API.");
        return Ok(());
    }

    println!("\nExecuting...");

    let runtime = InterpRuntime::new();
    let outputs = runtime.execute_snapshot(&snapshot, &[])?;

    println!("\nOutputs:");
    for (name, tensor) in outputs.iter() {
        println!("  {}: {:?} {:?}", name, tensor.dtype(), tensor.shape());
    }

    Ok(())
}
