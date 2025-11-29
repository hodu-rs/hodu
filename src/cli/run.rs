//! Run command - execute a .hdss model

use crate::common::{load_tensor, parse_device, parse_input_arg};
use crate::output::{self, OutputFormat};
use clap::Args;
use hodu_core::{format::hdss, script::Script};
use hodu_plugin::{Device, HoduError, HoduResult, InterpRuntime, PluginManager, RuntimePlugin, TensorData};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Args)]
pub struct RunArgs {
    /// Path to the .hdss file
    pub path: PathBuf,

    /// Device to run on (cpu, cuda:0, metal)
    #[arg(short, long, default_value = "cpu")]
    pub device: String,

    /// Input tensor file (format: name=path.hdt), can be repeated
    #[arg(short, long, action = clap::ArgAction::Append)]
    pub input: Vec<String>,

    /// Input tensor files, comma-separated (format: name=path.hdt,name=path.json)
    #[arg(short = 'I', long, value_delimiter = ',')]
    pub inputs: Vec<String>,

    /// Output format (pretty, json, hdt)
    #[arg(short = 'f', long, default_value = "pretty")]
    pub output_format: OutputFormat,

    /// Output directory for hdt format
    #[arg(short = 'o', long)]
    pub output_dir: Option<PathBuf>,

    /// Path to compiler plugin (.dylib/.so/.dll) for AOT compilation
    #[arg(long)]
    pub compiler_plugin: Option<PathBuf>,

    /// Path to runtime plugin (.dylib/.so/.dll) for execution
    #[arg(long)]
    pub runtime_plugin: Option<PathBuf>,
}

pub fn execute(args: RunArgs) -> HoduResult<()> {
    let device = parse_device(&args.device)?;
    let snapshot = hdss::load(&args.path)?;

    // Load input tensors
    let input_map = load_inputs(&args.input, &args.inputs, &snapshot)?;

    // Convert to TensorData bindings
    let input_data = prepare_input_data(&snapshot, &input_map)?;
    let input_bindings: Vec<(&str, TensorData)> = input_data
        .iter()
        .map(|(name, data)| (name.as_str(), data.clone()))
        .collect();

    // Execute
    let outputs = run_model(device, &snapshot, &input_bindings, &args)?;

    // Output results
    let target_order: Vec<String> = snapshot.targets.iter().map(|t| t.name.clone()).collect();
    output::print_outputs(&outputs, &target_order, args.output_format, args.output_dir.as_deref())
}

/// Load and validate input tensors
fn load_inputs(
    input: &[String],
    inputs: &[String],
    snapshot: &hodu_core::script::Snapshot,
) -> HoduResult<HashMap<String, hodu_core::tensor::Tensor>> {
    let all_inputs: Vec<&str> = input.iter().chain(inputs.iter()).map(|s| s.as_str()).collect();

    if all_inputs.is_empty() && !snapshot.inputs.is_empty() {
        return Err(HoduError::InvalidArgument(format!(
            "Missing inputs. Required: {}",
            snapshot
                .inputs
                .iter()
                .map(|i| format!("{} ({:?} {:?})", i.name, i.dtype, i.shape))
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }

    let mut input_map = HashMap::new();
    for s in all_inputs {
        let (name, path) = parse_input_arg(s)?;
        let tensor = load_tensor(&path)?;
        input_map.insert(name, tensor);
    }

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

    Ok(input_map)
}

/// Convert input tensors to TensorData format
fn prepare_input_data(
    snapshot: &hodu_core::script::Snapshot,
    input_map: &HashMap<String, hodu_core::tensor::Tensor>,
) -> HoduResult<Vec<(String, TensorData)>> {
    snapshot
        .inputs
        .iter()
        .map(|input| {
            let tensor = input_map.get(&input.name).unwrap();
            let data = tensor.to_bytes()?;
            let shape = tensor.shape().dims().to_vec();
            let dtype = tensor.dtype();
            Ok((input.name.clone(), TensorData::new(data, shape, dtype)))
        })
        .collect()
}

/// Run model on the specified device
fn run_model(
    device: Device,
    snapshot: &hodu_core::script::Snapshot,
    input_bindings: &[(&str, TensorData)],
    args: &RunArgs,
) -> HoduResult<HashMap<String, TensorData>> {
    match device {
        Device::CPU if args.compiler_plugin.is_some() => run_with_plugins(device, snapshot, input_bindings, args),
        Device::CPU => run_with_interpreter(snapshot, input_bindings),
        Device::Metal => run_with_plugins(device, snapshot, input_bindings, args),
        #[cfg(feature = "cuda")]
        Device::CUDA(_) => Err(HoduError::UnsupportedDevice(device)),
        #[allow(unreachable_patterns)]
        _ => Err(HoduError::UnsupportedDevice(device)),
    }
}

/// Run using interpreter (CPU only, no plugins)
fn run_with_interpreter(
    snapshot: &hodu_core::script::Snapshot,
    input_bindings: &[(&str, TensorData)],
) -> HoduResult<HashMap<String, TensorData>> {
    let runtime = InterpRuntime::new();
    let snapshot_data = snapshot.serialize()?;
    let artifact =
        hodu_plugin::CompiledArtifact::new(hodu_plugin::OutputFormat::HoduSnapshot, Device::CPU, snapshot_data);
    let module = runtime.load(&artifact, Device::CPU)?;
    module.execute(input_bindings)
}

/// Run using compiler and runtime plugins
fn run_with_plugins(
    device: Device,
    snapshot: &hodu_core::script::Snapshot,
    input_bindings: &[(&str, TensorData)],
    args: &RunArgs,
) -> HoduResult<HashMap<String, TensorData>> {
    let compiler_path = args
        .compiler_plugin
        .as_ref()
        .ok_or_else(|| HoduError::InvalidArgument(format!("{:?} device requires --compiler-plugin path", device)))?;

    let mut manager = PluginManager::with_default_dir()?;
    manager.load_compiler(compiler_path)?;

    let compiler = manager
        .compilers()
        .find(|c| c.supports_device(device))
        .ok_or_else(|| HoduError::BackendError(format!("No compiler found for {:?}", device)))?;

    let script = Script::new(snapshot.clone());
    let artifact = compiler.compile(&script, device)?;

    // Load runtime if provided
    if let Some(runtime_path) = &args.runtime_plugin {
        manager.load_runtime(runtime_path)?;
    }

    // Find appropriate runtime
    let runtime = manager
        .runtimes()
        .find(|r| r.supports_device(device) && r.loadable_formats(device).iter().any(|f| *f == artifact.format))
        .ok_or_else(|| {
            HoduError::BackendError(format!(
                "No runtime found for {:?} supporting {:?} format",
                device, artifact.format
            ))
        })?;

    let module = runtime.load(&artifact, device)?;
    module.execute(input_bindings)
}
