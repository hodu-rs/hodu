//! Plugin command - manage plugins (install, remove, list, etc.)
//!
//! This command manages JSON-RPC based plugins as standalone executables.

use crate::output;
use crate::plugins::{
    detect_plugin_type, DetectedPluginType, PluginCapabilities, PluginEntry, PluginRegistry, PluginSource, PluginType,
};
use clap::{Args, Subcommand};
use hodu_plugin_sdk::SDK_VERSION;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Args)]
pub struct PluginArgs {
    #[command(subcommand)]
    pub command: PluginCommands,
}

#[derive(Subcommand)]
pub enum PluginCommands {
    /// List installed plugins
    List,

    /// Show plugin info (spawns plugin to get runtime info)
    Info(InfoArgs),

    /// Install a plugin
    Install(InstallArgs),

    /// Remove a plugin
    Remove(RemoveArgs),

    /// Update plugins
    Update(UpdateArgs),

    /// Enable a plugin
    Enable(EnableArgs),

    /// Disable a plugin
    Disable(DisableArgs),

    /// Verify plugin integrity (check binaries exist, dependencies satisfied)
    Verify,

    /// Create a new plugin project
    Create(CreateArgs),
}

#[derive(Args)]
pub struct InfoArgs {
    /// Plugin name
    pub name: String,
}

#[derive(Args)]
pub struct InstallArgs {
    /// Plugin name (from official registry)
    pub name: Option<String>,

    /// Install from local path
    #[arg(long)]
    pub path: Option<PathBuf>,

    /// Install from git repository
    #[arg(long)]
    pub git: Option<String>,

    /// Subdirectory in git repository
    #[arg(long)]
    pub subdir: Option<String>,

    /// Git tag or branch
    #[arg(long)]
    pub tag: Option<String>,

    /// Force reinstall
    #[arg(long)]
    pub force: bool,

    /// Debug build
    #[arg(long)]
    pub debug: bool,
}

#[derive(Args)]
pub struct RemoveArgs {
    /// Plugin name
    pub name: String,
}

#[derive(Args)]
pub struct UpdateArgs {
    /// Plugin name (update all if not specified)
    pub name: Option<String>,
}

#[derive(Args)]
pub struct EnableArgs {
    /// Plugin name
    pub name: String,
}

#[derive(Args)]
pub struct DisableArgs {
    /// Plugin name
    pub name: String,
}

#[derive(Args)]
pub struct CreateArgs {
    /// Plugin name (e.g., hodu-backend-mybackend)
    pub name: String,

    /// Plugin type: backend, model_format, or tensor_format
    #[arg(long = "type", short = 't', default_value = "backend")]
    pub plugin_type: String,

    /// Output directory (default: current directory)
    #[arg(long, short = 'o')]
    pub output: Option<PathBuf>,
}

pub fn execute(args: PluginArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        PluginCommands::List => list_plugins(),
        PluginCommands::Info(info_args) => info_plugin(info_args),
        PluginCommands::Install(install_args) => install_plugin(install_args),
        PluginCommands::Remove(remove_args) => remove_plugin(remove_args),
        PluginCommands::Update(update_args) => update_plugins(update_args),
        PluginCommands::Enable(enable_args) => enable_plugin(enable_args),
        PluginCommands::Disable(disable_args) => disable_plugin(disable_args),
        PluginCommands::Verify => verify_plugins(),
        PluginCommands::Create(create_args) => create_plugin(create_args),
    }
}

fn list_plugins() -> Result<(), Box<dyn std::error::Error>> {
    let registry_path = PluginRegistry::default_path()?;
    let registry = PluginRegistry::load(&registry_path)?;

    // Backend plugins
    println!("Backend plugins:");
    let backends: Vec<_> = registry.all_backends().collect();
    if backends.is_empty() {
        println!("  (none installed)");
    } else {
        for plugin in backends {
            let caps = &plugin.capabilities;
            let mut features = Vec::new();
            if caps.runner.unwrap_or(false) {
                features.push("Runner");
            }
            if caps.builder.unwrap_or(false) {
                features.push("Builder");
            }
            let features_str = if features.is_empty() {
                String::new()
            } else {
                format!("[{}]", features.join(", "))
            };

            let devices = if caps.devices.is_empty() {
                String::new()
            } else {
                caps.devices.join(", ")
            };

            let status = if plugin.enabled { "" } else { " (disabled)" };

            println!(
                "  {:<20} {:<10} {:<18} {:<10} {}{}",
                plugin.name, plugin.version, features_str, devices, plugin.source, status
            );
        }
    }

    println!();

    // Model format plugins
    println!("Model format plugins:");
    let model_formats: Vec<_> = registry.all_model_formats().collect();
    if model_formats.is_empty() {
        println!("  (none installed)");
    } else {
        for plugin in model_formats {
            let caps = &plugin.capabilities;
            let mut features = Vec::new();
            if caps.load_model.unwrap_or(false) {
                features.push("load");
            }
            if caps.save_model.unwrap_or(false) {
                features.push("save");
            }
            let features_str = if features.is_empty() {
                String::new()
            } else {
                format!("[{}]", features.join(", "))
            };

            let extensions = if caps.model_extensions.is_empty() {
                String::new()
            } else {
                caps.model_extensions.join(", ")
            };

            let status = if plugin.enabled { "" } else { " (disabled)" };

            println!(
                "  {:<20} {:<10} {:<18} {:<15} {}{}",
                plugin.name, plugin.version, features_str, extensions, plugin.source, status
            );
        }
    }

    println!();

    // Tensor format plugins
    println!("Tensor format plugins:");
    let tensor_formats: Vec<_> = registry.all_tensor_formats().collect();
    if tensor_formats.is_empty() {
        println!("  (none installed)");
    } else {
        for plugin in tensor_formats {
            let caps = &plugin.capabilities;
            let mut features = Vec::new();
            if caps.load_tensor.unwrap_or(false) {
                features.push("load");
            }
            if caps.save_tensor.unwrap_or(false) {
                features.push("save");
            }
            let features_str = if features.is_empty() {
                String::new()
            } else {
                format!("[{}]", features.join(", "))
            };

            let extensions = if caps.tensor_extensions.is_empty() {
                String::new()
            } else {
                caps.tensor_extensions.join(", ")
            };

            let status = if plugin.enabled { "" } else { " (disabled)" };

            println!(
                "  {:<20} {:<10} {:<18} {:<15} {}{}",
                plugin.name, plugin.version, features_str, extensions, plugin.source, status
            );
        }
    }

    Ok(())
}

fn info_plugin(args: InfoArgs) -> Result<(), Box<dyn std::error::Error>> {
    use crate::plugins::PluginManager;

    let registry_path = PluginRegistry::default_path()?;
    let registry = PluginRegistry::load(&registry_path)?;

    // Find plugin
    let plugin = registry.find(&args.name).or_else(|| {
        let backend_name = format!("hodu-backend-{}", args.name);
        let format_name = format!("hodu-format-{}", args.name);
        registry.find(&backend_name).or_else(|| registry.find(&format_name))
    });

    let plugin = match plugin {
        Some(p) => p,
        None => return Err(format!("Plugin '{}' not found.", args.name).into()),
    };

    // Print registry info
    println!("Plugin: {}", plugin.name);
    println!("Version: {}", plugin.version);
    if let Some(desc) = &plugin.description {
        println!("Description: {}", desc);
    }
    if let Some(lic) = &plugin.license {
        println!("License: {}", lic);
    }
    println!("Type: {:?}", plugin.plugin_type);
    println!("Source: {}", plugin.source);
    println!("SDK Version: {}", plugin.sdk_version);
    println!("Installed: {}", plugin.installed_at);
    println!("Enabled: {}", plugin.enabled);
    if !plugin.dependencies.is_empty() {
        println!("Dependencies: {}", plugin.dependencies.join(", "));
    }
    println!();

    // Spawn plugin to get runtime info
    println!("Runtime info (from plugin):");
    let mut manager = PluginManager::new()?;
    let _client = manager.get_plugin(&plugin.name)?;

    if let Some(info) = manager.get_info(&plugin.name) {
        println!("  Name: {}", info.name);
        println!("  Version: {}", info.version);
        println!("  Capabilities: {}", info.capabilities.join(", "));
        if let Some(devices) = &info.devices {
            if !devices.is_empty() {
                println!("  Devices: {}", devices.join(", "));
            }
        }
        if let Some(extensions) = &info.model_extensions {
            if !extensions.is_empty() {
                println!("  Model extensions: {}", extensions.join(", "));
            }
        }
        if let Some(extensions) = &info.tensor_extensions {
            if !extensions.is_empty() {
                println!("  Tensor extensions: {}", extensions.join(", "));
            }
        }
    }

    Ok(())
}

fn install_plugin(args: InstallArgs) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = &args.path {
        let source = PluginSource::Local {
            path: path.canonicalize()?.to_string_lossy().to_string(),
        };
        install_from_path(path, args.debug, args.force, source)
    } else if let Some(git) = &args.git {
        install_from_git(git, args.subdir.as_deref(), args.tag.as_deref(), args.debug, args.force)
    } else if let Some(name) = &args.name {
        install_from_registry(name, args.tag.as_deref(), args.debug, args.force)
    } else {
        Err("No plugin specified. Use <name>, --path, or --git.".into())
    }
}

/// Official plugin registry URL
const PLUGIN_REGISTRY_URL: &str = "https://raw.githubusercontent.com/daminstudio/hodu-plugins/main/plugins.toml";

/// Version entry in the registry
#[derive(Debug, serde::Deserialize)]
struct PluginVersionEntry {
    version: String,
    tag: String,
    /// SDK version requirement (e.g., "0.1" means compatible with 0.1.x)
    sdk: String,
}

/// Plugin entry in the registry
#[derive(Debug, serde::Deserialize)]
struct RegistryPlugin {
    name: String,
    #[allow(dead_code)]
    description: Option<String>,
    git: String,
    path: Option<String>,
    versions: Vec<PluginVersionEntry>,
}

/// Registry file structure
#[derive(Debug, serde::Deserialize)]
struct PluginRegistryFile {
    plugin: Vec<RegistryPlugin>,
}

fn install_from_registry(
    name_with_version: &str,
    tag_override: Option<&str>,
    debug: bool,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Parse name@version syntax
    let (name, requested_version) = if let Some(at_pos) = name_with_version.find('@') {
        let n = &name_with_version[..at_pos];
        let v = &name_with_version[at_pos + 1..];
        (n, Some(v))
    } else {
        (name_with_version, None)
    };

    output::fetching(&format!("plugin registry for '{}'", name));

    // Fetch registry
    let body = ureq::get(PLUGIN_REGISTRY_URL)
        .call()
        .map_err(|e| format!("Failed to fetch plugin registry: {}", e))?
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("Failed to read registry: {}", e))?;

    let registry: PluginRegistryFile = toml::from_str(&body).map_err(|e| format!("Failed to parse registry: {}", e))?;

    // Find plugin
    let plugin = registry.plugin.iter().find(|p| p.name == name).ok_or_else(|| {
        let available: Vec<_> = registry.plugin.iter().map(|p| p.name.as_str()).collect();
        format!(
            "Plugin '{}' not found in registry.\n\nAvailable plugins:\n  {}",
            name,
            available.join("\n  ")
        )
    })?;

    // Get host SDK version (major.minor)
    let host_sdk_parts: Vec<&str> = SDK_VERSION.split('.').collect();
    let host_sdk_major_minor = if host_sdk_parts.len() >= 2 {
        format!("{}.{}", host_sdk_parts[0], host_sdk_parts[1])
    } else {
        SDK_VERSION.to_string()
    };

    // Filter compatible versions (same major.minor)
    let compatible_versions: Vec<_> = plugin
        .versions
        .iter()
        .filter(|v| v.sdk == host_sdk_major_minor)
        .collect();

    // Determine the tag to use
    let tag = if let Some(t) = tag_override {
        // --tag flag takes precedence
        Some(t.to_string())
    } else if let Some(ver) = requested_version {
        // @version syntax
        if ver == "latest" {
            // Use the first compatible version (latest)
            if compatible_versions.is_empty() {
                return Err(format!(
                    "No compatible version found for SDK {}.\n\nAvailable versions:\n  {}",
                    host_sdk_major_minor,
                    plugin
                        .versions
                        .iter()
                        .map(|v| format!("{} (sdk {})", v.version, v.sdk))
                        .collect::<Vec<_>>()
                        .join("\n  ")
                )
                .into());
            }
            compatible_versions.first().map(|v| v.tag.clone())
        } else {
            // Find specific version
            let version_entry = plugin.versions.iter().find(|v| v.version == ver).ok_or_else(|| {
                let available: Vec<_> = plugin.versions.iter().map(|v| v.version.as_str()).collect();
                format!(
                    "Version '{}' not found for plugin '{}'.\n\nAvailable versions:\n  {}",
                    ver,
                    name,
                    available.join("\n  ")
                )
            })?;
            Some(version_entry.tag.clone())
        }
    } else {
        // No version specified, use latest compatible
        if compatible_versions.is_empty() {
            return Err(format!(
                "No compatible version found for SDK {}.\n\nAvailable versions:\n  {}",
                host_sdk_major_minor,
                plugin
                    .versions
                    .iter()
                    .map(|v| format!("{} (sdk {})", v.version, v.sdk))
                    .collect::<Vec<_>>()
                    .join("\n  ")
            )
            .into());
        }
        compatible_versions.first().map(|v| v.tag.clone())
    };

    install_from_git(&plugin.git, plugin.path.as_deref(), tag.as_deref(), debug, force)
}

fn install_from_git(
    url: &str,
    subdir: Option<&str>,
    tag: Option<&str>,
    debug: bool,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create temp directory
    let temp_dir = std::env::temp_dir().join(format!("hodu_plugin_{}", std::process::id()));
    if temp_dir.exists() {
        std::fs::remove_dir_all(&temp_dir)?;
    }

    // Clone repository (quietly)
    let mut git_cmd = Command::new("git");
    git_cmd.arg("clone").arg("-q");
    if tag.is_none() {
        git_cmd.arg("--depth").arg("1");
    }
    git_cmd.arg(url).arg(&temp_dir);

    let status = git_cmd.status()?;
    if !status.success() {
        return Err(format!("Failed to clone repository: {}", url).into());
    }

    // Checkout tag/branch if specified (quietly)
    if let Some(t) = tag {
        let status = Command::new("git")
            .arg("checkout")
            .arg("-q")
            .arg(t)
            .current_dir(&temp_dir)
            .status()?;
        if !status.success() {
            std::fs::remove_dir_all(&temp_dir)?;
            return Err(format!("Failed to checkout: {}", t).into());
        }
    }

    // Determine install path (with optional subdir)
    let install_path = match subdir {
        Some(s) => temp_dir.join(s),
        None => temp_dir.clone(),
    };

    if !install_path.exists() {
        std::fs::remove_dir_all(&temp_dir)?;
        return Err(format!("Subdirectory '{}' not found in repository", subdir.unwrap_or("")).into());
    }

    // Install from the cloned path
    let source = PluginSource::Git {
        url: url.to_string(),
        tag: tag.map(|t| t.to_string()),
        subdir: subdir.map(|s| s.to_string()),
    };
    let result = install_from_path(&install_path, debug, force, source);

    // Cleanup temp directory
    let _ = std::fs::remove_dir_all(&temp_dir);

    result
}

fn install_from_path(
    path: &Path,
    debug: bool,
    force: bool,
    source: PluginSource,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = path.canonicalize()?;

    // Check if it's a Cargo project
    let cargo_toml = path.join("Cargo.toml");
    if !cargo_toml.exists() {
        return Err(format!("No Cargo.toml found at {}", path.display()).into());
    }

    // Parse Cargo.toml to get the package name
    let cargo_content = std::fs::read_to_string(&cargo_toml)?;
    let package_name = parse_package_name(&cargo_content)
        .ok_or_else(|| format!("Could not find package name in {}", cargo_toml.display()))?;

    // Build the plugin with cargo (as executable)
    output::compiling(&format!("{}", package_name));
    let mut cargo_cmd = Command::new("cargo");
    cargo_cmd.arg("build");
    cargo_cmd.arg("-p").arg(&package_name);
    if !debug {
        cargo_cmd.arg("--release");
    }
    cargo_cmd.arg("-q"); // Quiet output
    cargo_cmd.current_dir(&path);

    let cmd_output = cargo_cmd.output()?;
    if !cmd_output.status.success() {
        output::error("build failed");
        return Err(format!(
            "Failed to build plugin:\n{}",
            String::from_utf8_lossy(&cmd_output.stderr)
        )
        .into());
    }

    // Find the built executable
    let profile = if debug { "debug" } else { "release" };

    // Try multiple possible target directories
    let possible_target_dirs = vec![
        path.join("target").join(profile),
        path.parent()
            .map(|p| p.join("target").join(profile))
            .unwrap_or_default(),
    ];

    let mut bin_path = None;
    for target_dir in &possible_target_dirs {
        if !target_dir.exists() {
            continue;
        }

        // Look for executable matching the package name
        let candidate = target_dir.join(&package_name);
        if candidate.exists() {
            bin_path = Some(candidate);
            break;
        }

        // On Windows, add .exe
        #[cfg(windows)]
        {
            let candidate = target_dir.join(format!("{}.exe", package_name));
            if candidate.exists() {
                bin_path = Some(candidate);
                break;
            }
        }
    }

    let bin_path = bin_path.ok_or_else(|| {
        format!(
            "No executable found for package '{}'. Checked: {:?}",
            package_name, possible_target_dirs
        )
    })?;

    // Read manifest.json if it exists, or detect from binary
    let manifest_path = path.join("manifest.json");
    let (name, version, sdk_version, plugin_type, capabilities) = if manifest_path.exists() {
        // Read from manifest
        let manifest_content = std::fs::read_to_string(&manifest_path)?;
        let manifest: serde_json::Value = serde_json::from_str(&manifest_content)?;

        let name = manifest["name"].as_str().unwrap_or(&package_name).to_string();
        let version = manifest["version"].as_str().unwrap_or("0.1.0").to_string();
        let sdk_version = manifest["sdk_version"].as_str().unwrap_or(SDK_VERSION).to_string();

        let caps = manifest["capabilities"].as_array();
        let is_backend = caps
            .map(|c| {
                c.iter()
                    .any(|v| v.as_str().map(|s| s.starts_with("backend.")).unwrap_or(false))
            })
            .unwrap_or(false);

        // Determine plugin type and capabilities from manifest
        let has_model_caps = caps
            .map(|c| {
                c.iter().any(|v| {
                    v.as_str()
                        .map(|s| s == "format.load_model" || s == "format.save_model")
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);
        let has_tensor_caps = caps
            .map(|c| {
                c.iter().any(|v| {
                    v.as_str()
                        .map(|s| s == "format.load_tensor" || s == "format.save_tensor")
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);

        let extensions: Vec<String> = manifest["extensions"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let (plugin_type, capabilities) = if is_backend {
            let devices: Vec<String> = manifest["devices"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let runner = caps
                .map(|c| c.iter().any(|v| v.as_str() == Some("backend.run")))
                .unwrap_or(false);
            let builder = caps
                .map(|c| c.iter().any(|v| v.as_str() == Some("backend.build")))
                .unwrap_or(false);
            (
                PluginType::Backend,
                PluginCapabilities::backend(runner, builder, devices, vec![]),
            )
        } else if has_model_caps {
            let load_model = caps
                .map(|c| c.iter().any(|v| v.as_str() == Some("format.load_model")))
                .unwrap_or(false);
            let save_model = caps
                .map(|c| c.iter().any(|v| v.as_str() == Some("format.save_model")))
                .unwrap_or(false);
            (
                PluginType::ModelFormat,
                PluginCapabilities::model_format(load_model, save_model, extensions),
            )
        } else if has_tensor_caps {
            let load_tensor = caps
                .map(|c| c.iter().any(|v| v.as_str() == Some("format.load_tensor")))
                .unwrap_or(false);
            let save_tensor = caps
                .map(|c| c.iter().any(|v| v.as_str() == Some("format.save_tensor")))
                .unwrap_or(false);
            (
                PluginType::TensorFormat,
                PluginCapabilities::tensor_format(load_tensor, save_tensor, extensions),
            )
        } else {
            // Default to ModelFormat
            (
                PluginType::ModelFormat,
                PluginCapabilities::model_format(true, false, extensions),
            )
        };

        (name, version, sdk_version, plugin_type, capabilities)
    } else {
        // Try to detect from binary (spawn and initialize)
        let detected = detect_plugin_type(&bin_path)?;
        match detected {
            DetectedPluginType::Backend {
                name,
                version,
                sdk_version,
            } => {
                let capabilities = PluginCapabilities::backend(true, false, vec![], vec![]);
                (name, version, sdk_version, PluginType::Backend, capabilities)
            },
            DetectedPluginType::ModelFormat {
                name,
                version,
                sdk_version,
            } => {
                let capabilities = PluginCapabilities::model_format(true, false, vec![]);
                (name, version, sdk_version, PluginType::ModelFormat, capabilities)
            },
            DetectedPluginType::TensorFormat {
                name,
                version,
                sdk_version,
            } => {
                let capabilities = PluginCapabilities::tensor_format(true, false, vec![]);
                (name, version, sdk_version, PluginType::TensorFormat, capabilities)
            },
        }
    };

    // Check SDK version compatibility
    let host_parts: Vec<u32> = SDK_VERSION.split('.').filter_map(|s| s.parse().ok()).collect();
    let plugin_parts: Vec<u32> = sdk_version.split('.').filter_map(|s| s.parse().ok()).collect();

    if host_parts.len() >= 2 && plugin_parts.len() >= 2 {
        let (host_major, host_minor) = (host_parts[0], host_parts[1]);
        let (plugin_major, plugin_minor) = (plugin_parts[0], plugin_parts[1]);

        if host_major != plugin_major {
            return Err(format!(
                "SDK major version mismatch: host={}, plugin={}",
                SDK_VERSION, sdk_version
            )
            .into());
        }

        if host_minor < plugin_minor {
            return Err(format!(
                "Plugin requires newer SDK: host={}, plugin={}",
                SDK_VERSION, sdk_version
            )
            .into());
        }
    }

    // Load registry
    let registry_path = PluginRegistry::default_path()?;
    let mut registry = PluginRegistry::load(&registry_path)?;

    // Check if already installed
    if let Some(existing) = registry.find(&name) {
        if !force {
            return Err(format!(
                "Plugin {} v{} is already installed. Use --force to reinstall.",
                existing.name, existing.version
            )
            .into());
        }
    }

    // Copy executable to plugins directory
    let plugins_dir = get_plugins_dir()?;
    let plugin_dir = plugins_dir.join(&name);
    std::fs::create_dir_all(&plugin_dir)?;

    let bin_filename = bin_path.file_name().unwrap().to_string_lossy().to_string();
    let dest_path = plugin_dir.join(&bin_filename);
    std::fs::copy(&bin_path, &dest_path)?;

    // Make executable on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&dest_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&dest_path, perms)?;
    }

    // Parse metadata from manifest if available
    let (description, license, dependencies) = if manifest_path.exists() {
        let manifest_content = std::fs::read_to_string(&manifest_path)?;
        let manifest: serde_json::Value = serde_json::from_str(&manifest_content)?;
        let desc = manifest["description"].as_str().map(String::from);
        let lic = manifest["license"].as_str().map(String::from);
        let deps = manifest["dependencies"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        (desc, lic, deps)
    } else {
        (None, None, Vec::new())
    };

    // Create registry entry
    let entry = PluginEntry {
        name: name.clone(),
        version: version.clone(),
        description,
        license,
        plugin_type,
        capabilities,
        binary: bin_filename,
        source,
        installed_at: chrono_now(),
        sdk_version,
        enabled: true,
        dependencies: dependencies.clone(),
    };

    // Update registry
    registry.upsert(entry);
    registry.save(&registry_path)?;

    // Check dependencies after installation
    if !dependencies.is_empty() {
        if let Err(missing) = registry.check_dependencies(&name) {
            eprintln!("Warning: Missing dependencies for {}: {}", name, missing.join(", "));
        }
    }

    output::installed(&format!("{} v{}", name, version));
    Ok(())
}

fn chrono_now() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn remove_plugin(args: RemoveArgs) -> Result<(), Box<dyn std::error::Error>> {
    let registry_path = PluginRegistry::default_path()?;
    let mut registry = PluginRegistry::load(&registry_path)?;

    // Find the plugin
    let plugin = registry.find(&args.name);
    if plugin.is_none() {
        let backend_name = format!("hodu-backend-{}", args.name);
        let format_name = format!("hodu-format-{}", args.name);

        if registry.find(&backend_name).is_some() {
            return remove_plugin(RemoveArgs { name: backend_name });
        } else if registry.find(&format_name).is_some() {
            return remove_plugin(RemoveArgs { name: format_name });
        }

        return Err(format!(
            "Plugin '{}' not found.\n\nInstalled plugins:\n{}",
            args.name,
            list_installed_plugins(&registry)
        )
        .into());
    }

    let plugin = plugin.unwrap();
    let name = plugin.name.clone();
    let version = plugin.version.clone();

    // Delete the plugin directory
    let plugins_dir = get_plugins_dir()?;
    let plugin_dir = plugins_dir.join(&name);

    output::removing(&format!("{} v{}", name, version));

    if plugin_dir.exists() {
        std::fs::remove_dir_all(&plugin_dir)?;
    }

    // Remove from registry
    registry.remove(&name);
    registry.save(&registry_path)?;

    output::removed(&format!("{} v{}", name, version));
    Ok(())
}

fn enable_plugin(args: EnableArgs) -> Result<(), Box<dyn std::error::Error>> {
    let registry_path = PluginRegistry::default_path()?;
    let mut registry = PluginRegistry::load(&registry_path)?;

    // Try to find plugin with various name formats
    let name = find_plugin_name(&registry, &args.name)?;

    if registry.enable(&name) {
        registry.save(&registry_path)?;
        output::finished(&format!("enabled {}", name));
    } else {
        return Err(format!("Plugin '{}' not found.", args.name).into());
    }

    Ok(())
}

fn disable_plugin(args: DisableArgs) -> Result<(), Box<dyn std::error::Error>> {
    let registry_path = PluginRegistry::default_path()?;
    let mut registry = PluginRegistry::load(&registry_path)?;

    // Try to find plugin with various name formats
    let name = find_plugin_name(&registry, &args.name)?;

    // Check if other plugins depend on this one
    let dependents: Vec<String> = registry
        .plugins
        .iter()
        .filter(|p| p.enabled && p.dependencies.contains(&name))
        .map(|p| p.name.clone())
        .collect();

    if !dependents.is_empty() {
        return Err(format!("Cannot disable '{}': required by {}", name, dependents.join(", ")).into());
    }

    if registry.disable(&name) {
        registry.save(&registry_path)?;
        output::finished(&format!("disabled {}", name));
    } else {
        return Err(format!("Plugin '{}' not found.", args.name).into());
    }

    Ok(())
}

fn verify_plugins() -> Result<(), Box<dyn std::error::Error>> {
    let registry_path = PluginRegistry::default_path()?;
    let registry = PluginRegistry::load(&registry_path)?;
    let plugins_dir = get_plugins_dir()?;

    let mut issues = Vec::new();
    let mut ok_count = 0;

    for plugin in &registry.plugins {
        let mut plugin_issues = Vec::new();

        // Check if binary exists
        let binary_path = plugins_dir.join(&plugin.name).join(&plugin.binary);
        if !binary_path.exists() {
            plugin_issues.push(format!("binary not found: {}", binary_path.display()));
        }

        // Check dependencies (only for enabled plugins)
        if plugin.enabled {
            let missing_deps: Vec<&String> = plugin
                .dependencies
                .iter()
                .filter(|dep| registry.find(dep).map(|p| !p.enabled).unwrap_or(true))
                .collect();

            if !missing_deps.is_empty() {
                plugin_issues.push(format!(
                    "missing dependencies: {}",
                    missing_deps.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                ));
            }
        }

        if plugin_issues.is_empty() {
            ok_count += 1;
        } else {
            let status = if plugin.enabled { "" } else { " (disabled)" };
            issues.push(format!("  {}{}: {}", plugin.name, status, plugin_issues.join("; ")));
        }
    }

    if issues.is_empty() {
        println!("All {} plugins verified OK.", ok_count);
    } else {
        println!(
            "Verified {} plugins, {} with issues:",
            ok_count + issues.len(),
            issues.len()
        );
        for issue in issues {
            println!("{}", issue);
        }
    }

    Ok(())
}

fn find_plugin_name(registry: &PluginRegistry, name: &str) -> Result<String, Box<dyn std::error::Error>> {
    if registry.find(name).is_some() {
        return Ok(name.to_string());
    }

    let backend_name = format!("hodu-backend-{}", name);
    if registry.find(&backend_name).is_some() {
        return Ok(backend_name);
    }

    let format_name = format!("hodu-format-{}", name);
    if registry.find(&format_name).is_some() {
        return Ok(format_name);
    }

    Err(format!("Plugin '{}' not found.", name).into())
}

fn list_installed_plugins(registry: &PluginRegistry) -> String {
    let mut result = String::new();

    let backends: Vec<_> = registry.backends().collect();
    if !backends.is_empty() {
        result.push_str("  Backend: ");
        result.push_str(&backends.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", "));
        result.push('\n');
    }

    let model_formats: Vec<_> = registry.model_formats().collect();
    if !model_formats.is_empty() {
        result.push_str("  Model format: ");
        result.push_str(
            &model_formats
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        );
        result.push('\n');
    }

    let tensor_formats: Vec<_> = registry.tensor_formats().collect();
    if !tensor_formats.is_empty() {
        result.push_str("  Tensor format: ");
        result.push_str(
            &tensor_formats
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        );
        result.push('\n');
    }

    if result.is_empty() {
        result.push_str("  (none installed)\n");
    }

    result
}

fn update_plugins(args: UpdateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let registry_path = PluginRegistry::default_path()?;
    let registry = PluginRegistry::load(&registry_path)?;

    let plugins_to_update: Vec<_> = if let Some(name) = &args.name {
        // Update specific plugin
        match registry.find(name) {
            Some(p) => vec![p.clone()],
            None => {
                // Try with prefix
                let backend_name = format!("hodu-backend-{}", name);
                let format_name = format!("hodu-format-{}", name);
                if let Some(p) = registry.find(&backend_name) {
                    vec![p.clone()]
                } else if let Some(p) = registry.find(&format_name) {
                    vec![p.clone()]
                } else {
                    return Err(format!("Plugin '{}' not found.", name).into());
                }
            },
        }
    } else {
        // Update all plugins
        registry.plugins.clone()
    };

    if plugins_to_update.is_empty() {
        println!("No plugins to update.");
        return Ok(());
    }

    // Try to fetch the official registry for version info
    let official_registry = fetch_official_registry().ok();

    for plugin in plugins_to_update {
        output::updating(&format!("{}", plugin.name));

        // Check if plugin is from official registry and has a newer version
        if let Some(ref reg) = official_registry {
            if let Some(reg_plugin) = reg.plugin.iter().find(|p| p.name == plugin.name) {
                // Get host SDK version
                let host_sdk_parts: Vec<&str> = SDK_VERSION.split('.').collect();
                let host_sdk = if host_sdk_parts.len() >= 2 {
                    format!("{}.{}", host_sdk_parts[0], host_sdk_parts[1])
                } else {
                    SDK_VERSION.to_string()
                };

                // Find latest compatible version
                if let Some(latest) = reg_plugin.versions.iter().find(|v| v.sdk == host_sdk) {
                    if latest.version != plugin.version {
                        println!("  {} -> {} (sdk {})", plugin.version, latest.version, latest.sdk);
                        install_from_registry(&plugin.name, None, false, true)?;
                        continue;
                    } else {
                        println!("  Already at latest compatible version: {}", plugin.version);
                        continue;
                    }
                }
            }
        }

        // Fallback to source-based update
        match &plugin.source {
            PluginSource::Git { url, tag, subdir } => {
                install_from_git(url, subdir.as_deref(), tag.as_deref(), false, true)?;
            },
            PluginSource::Local { path } => {
                let path_buf = PathBuf::from(path);
                if path_buf.exists() {
                    let source = PluginSource::Local { path: path.clone() };
                    install_from_path(&path_buf, false, true, source)?;
                } else {
                    println!("  Warning: Source path no longer exists: {}", path_buf.display());
                }
            },
            PluginSource::CratesIo => {
                println!("  Skipped: crates.io source (reinstall with --git or --path)");
            },
        }
    }

    Ok(())
}

fn fetch_official_registry() -> Result<PluginRegistryFile, Box<dyn std::error::Error>> {
    let body = ureq::get(PLUGIN_REGISTRY_URL)
        .call()
        .map_err(|e| format!("Failed to fetch plugin registry: {}", e))?
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("Failed to read registry: {}", e))?;

    let registry: PluginRegistryFile = toml::from_str(&body).map_err(|e| format!("Failed to parse registry: {}", e))?;
    Ok(registry)
}

fn get_plugins_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let home = dirs::home_dir().ok_or("Could not find home directory")?;
    let plugins_dir = home.join(".hodu").join("plugins");

    if !plugins_dir.exists() {
        std::fs::create_dir_all(&plugins_dir)?;
    }

    Ok(plugins_dir)
}

/// Parse package name from Cargo.toml content
fn parse_package_name(content: &str) -> Option<String> {
    let mut in_package = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[package]" {
            in_package = true;
            continue;
        }
        if trimmed.starts_with('[') && trimmed != "[package]" {
            in_package = false;
            continue;
        }
        if in_package && trimmed.starts_with("name") {
            if let Some(eq_pos) = trimmed.find('=') {
                let value = trimmed[eq_pos + 1..].trim();
                let value = value.trim_matches('"').trim_matches('\'');
                return Some(value.to_string());
            }
        }
    }
    None
}

fn create_plugin(args: CreateArgs) -> Result<(), Box<dyn std::error::Error>> {
    use hodu_plugin_sdk::{
        cargo_toml_template, main_rs_backend_template, main_rs_model_format_template, main_rs_tensor_format_template,
        manifest_json_backend_template, manifest_json_model_format_template, manifest_json_tensor_format_template,
    };

    let plugin_type = args.plugin_type.to_lowercase();
    let valid_types = ["backend", "model_format", "tensor_format"];
    if !valid_types.contains(&plugin_type.as_str()) {
        return Err(format!(
            "Invalid plugin type: '{}'. Use 'backend', 'model_format', or 'tensor_format'.",
            args.plugin_type
        )
        .into());
    }

    let output_dir = args.output.unwrap_or_else(|| std::env::current_dir().unwrap());
    let project_dir = output_dir.join(&args.name);

    if project_dir.exists() {
        return Err(format!("Directory already exists: {}", project_dir.display()).into());
    }

    println!("Creating {} plugin: {}", plugin_type, args.name);

    std::fs::create_dir_all(&project_dir)?;
    std::fs::create_dir_all(project_dir.join("src"))?;

    // Cargo.toml
    let cargo_toml = cargo_toml_template(&args.name);
    std::fs::write(project_dir.join("Cargo.toml"), cargo_toml)?;

    // manifest.json
    let manifest = match plugin_type.as_str() {
        "backend" => manifest_json_backend_template(&args.name),
        "model_format" => manifest_json_model_format_template(&args.name),
        "tensor_format" => manifest_json_tensor_format_template(&args.name),
        _ => unreachable!(),
    };
    std::fs::write(project_dir.join("manifest.json"), manifest)?;

    // main.rs
    let main_rs = match plugin_type.as_str() {
        "backend" => main_rs_backend_template(&args.name),
        "model_format" => main_rs_model_format_template(&args.name),
        "tensor_format" => main_rs_tensor_format_template(&args.name),
        _ => unreachable!(),
    };
    std::fs::write(project_dir.join("src").join("main.rs"), main_rs)?;

    println!("Created plugin project at: {}", project_dir.display());
    println!();
    println!("Next steps:");
    println!("  1. cd {}", args.name);
    println!("  2. Edit manifest.json with your plugin details");
    println!("  3. Implement the plugin in src/main.rs");
    println!("  4. Install with: hodu plugin install --path .");

    Ok(())
}
