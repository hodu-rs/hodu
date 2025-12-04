//! Plugin command - manage plugins (install, remove, list, etc.)
//!
//! This command manages JSON-RPC based plugins as standalone executables.

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
        PluginCommands::Create(create_args) => create_plugin(create_args),
    }
}

fn list_plugins() -> Result<(), Box<dyn std::error::Error>> {
    let registry_path = PluginRegistry::default_path()?;
    let registry = PluginRegistry::load(&registry_path)?;

    // Backend plugins
    println!("Backend plugins:");
    let backends: Vec<_> = registry.backends().collect();
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

            println!(
                "  {:<20} {:<10} {:<18} {:<10} {}",
                plugin.name, plugin.version, features_str, devices, plugin.source
            );
        }
    }

    println!();

    // Model format plugins
    println!("Model format plugins:");
    let model_formats: Vec<_> = registry.model_formats().collect();
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

            println!(
                "  {:<20} {:<10} {:<18} {:<15} {}",
                plugin.name, plugin.version, features_str, extensions, plugin.source
            );
        }
    }

    println!();

    // Tensor format plugins
    println!("Tensor format plugins:");
    let tensor_formats: Vec<_> = registry.tensor_formats().collect();
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

            println!(
                "  {:<20} {:<10} {:<18} {:<15} {}",
                plugin.name, plugin.version, features_str, extensions, plugin.source
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
    println!("Type: {:?}", plugin.plugin_type);
    println!("Source: {}", plugin.source);
    println!("SDK Version: {}", plugin.sdk_version);
    println!("Installed: {}", plugin.installed_at);
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
        let source = PluginSource::Local(path.canonicalize()?.to_string_lossy().to_string());
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

/// Plugin entry in the registry
#[derive(Debug, serde::Deserialize)]
struct RegistryPlugin {
    name: String,
    #[allow(dead_code)]
    description: Option<String>,
    git: String,
    path: Option<String>,
}

/// Registry file structure
#[derive(Debug, serde::Deserialize)]
struct PluginRegistryFile {
    plugin: Vec<RegistryPlugin>,
}

fn install_from_registry(
    name: &str,
    tag: Option<&str>,
    debug: bool,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Looking up '{}' in official plugin registry...", name);

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

    println!("Found: {} -> {}", plugin.name, plugin.git);

    install_from_git(&plugin.git, plugin.path.as_deref(), tag, debug, force)
}

fn install_from_git(
    url: &str,
    subdir: Option<&str>,
    tag: Option<&str>,
    debug: bool,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Installing from git: {}", url);
    if let Some(s) = subdir {
        println!("Subdirectory: {}", s);
    }
    if let Some(t) = tag {
        println!("Tag/branch: {}", t);
    }

    // Create temp directory
    let temp_dir = std::env::temp_dir().join(format!("hodu_plugin_{}", std::process::id()));
    if temp_dir.exists() {
        std::fs::remove_dir_all(&temp_dir)?;
    }

    // Clone repository
    println!("Cloning repository...");
    let mut git_cmd = Command::new("git");
    git_cmd.arg("clone");
    if tag.is_none() {
        git_cmd.arg("--depth").arg("1");
    }
    git_cmd.arg(url).arg(&temp_dir);

    let status = git_cmd.status()?;
    if !status.success() {
        return Err(format!("Failed to clone repository: {}", url).into());
    }

    // Checkout tag/branch if specified
    if let Some(t) = tag {
        println!("Checking out: {}", t);
        let status = Command::new("git")
            .arg("checkout")
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
    let source = PluginSource::Git(url.to_string());
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
    println!("Installing plugin from: {}", path.display());

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
    println!("Building plugin...");
    let mut cargo_cmd = Command::new("cargo");
    cargo_cmd.arg("build");
    cargo_cmd.arg("-p").arg(&package_name);
    if !debug {
        cargo_cmd.arg("--release");
    }
    cargo_cmd.current_dir(&path);

    let status = cargo_cmd.status()?;
    if !status.success() {
        return Err("Failed to build plugin".into());
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
    println!("Found executable: {}", bin_path.display());

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
                println!("Detected backend plugin: {} v{}", name, version);
                let capabilities = PluginCapabilities::backend(true, false, vec![], vec![]);
                (name, version, sdk_version, PluginType::Backend, capabilities)
            },
            DetectedPluginType::ModelFormat {
                name,
                version,
                sdk_version,
            } => {
                println!("Detected model format plugin: {} v{}", name, version);
                let capabilities = PluginCapabilities::model_format(true, false, vec![]);
                (name, version, sdk_version, PluginType::ModelFormat, capabilities)
            },
            DetectedPluginType::TensorFormat {
                name,
                version,
                sdk_version,
            } => {
                println!("Detected tensor format plugin: {} v{}", name, version);
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
        println!("Reinstalling {} (--force)", name);
    }

    // Copy executable to plugins directory
    let plugins_dir = get_plugins_dir()?;
    let plugin_dir = plugins_dir.join(&name);
    std::fs::create_dir_all(&plugin_dir)?;

    let bin_filename = bin_path.file_name().unwrap().to_string_lossy().to_string();
    let dest_path = plugin_dir.join(&bin_filename);

    println!("Copying to: {}", dest_path.display());
    std::fs::copy(&bin_path, &dest_path)?;

    // Make executable on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&dest_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&dest_path, perms)?;
    }

    // Create registry entry
    let entry = PluginEntry {
        name: name.clone(),
        version,
        plugin_type,
        capabilities,
        binary: bin_filename,
        source,
        installed_at: chrono_now(),
        sdk_version,
    };

    // Update registry
    registry.upsert(entry);
    registry.save(&registry_path)?;

    println!("Successfully installed plugin: {}", name);
    Ok(())
}

fn chrono_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("{}", now)
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

    if plugin_dir.exists() {
        std::fs::remove_dir_all(&plugin_dir)?;
        println!("Removed plugin directory: {}", plugin_dir.display());
    }

    // Remove from registry
    registry.remove(&name);
    registry.save(&registry_path)?;

    println!("Successfully removed plugin: {} v{}", name, version);
    Ok(())
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

    for plugin in plugins_to_update {
        println!("Updating {}...", plugin.name);
        match &plugin.source {
            PluginSource::Git(url) => {
                install_from_git(url, None, None, false, true)?;
            },
            PluginSource::Local(path) => {
                let path_buf = PathBuf::from(path);
                if path_buf.exists() {
                    let source = PluginSource::Local(path.clone());
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
    println!("  2. Implement the plugin in src/main.rs");
    println!("  3. Install with: hodu plugin install --path .");

    Ok(())
}
