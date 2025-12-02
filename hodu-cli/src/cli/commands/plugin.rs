use crate::cli::plugin::{
    detect_plugin_type, DetectedPluginType, LoadedBackendPlugin, LoadedFormatPlugin, PluginCapabilities, PluginEntry,
    PluginRegistry, PluginSource, PluginType,
};
use clap::{Args, Subcommand};
use hodu_cli_plugin_sdk::SDK_VERSION;
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

    /// Install a plugin
    Install(InstallArgs),

    /// Remove a plugin
    Remove(RemoveArgs),

    /// Update plugins
    Update(UpdateArgs),

    /// Search for plugins on crates.io
    Search(SearchArgs),

    /// Create a new plugin project
    Create(CreateArgs),
}

#[derive(Args)]
pub struct InstallArgs {
    /// Plugin name or name@version
    pub name: Option<String>,

    /// Install from local path
    #[arg(long)]
    pub path: Option<PathBuf>,

    /// Install from git repository
    #[arg(long)]
    pub git: Option<String>,

    /// Git tag or branch
    #[arg(long)]
    pub tag: Option<String>,

    /// Force reinstall
    #[arg(long)]
    pub force: bool,

    /// Debug build
    #[arg(long)]
    pub debug: bool,

    /// Install from lock file
    #[arg(long = "from")]
    pub from_lock: Option<PathBuf>,

    /// Install bundle (basic, apple, nvidia, dev)
    #[arg(long)]
    pub bundle: Option<String>,

    /// Trust unverified source
    #[arg(long)]
    pub trust: bool,
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
pub struct SearchArgs {
    /// Search query
    pub query: String,

    /// Filter by type (backend, format)
    #[arg(long = "type")]
    pub plugin_type: Option<String>,
}

#[derive(Args)]
pub struct CreateArgs {
    /// Plugin name (e.g., hodu-backend-mybackend)
    pub name: String,

    /// Plugin type: backend or format
    #[arg(long = "type", short = 't', default_value = "backend")]
    pub plugin_type: String,

    /// Output directory (default: current directory)
    #[arg(long, short = 'o')]
    pub output: Option<PathBuf>,
}

pub fn execute(args: PluginArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        PluginCommands::List => list_plugins(),
        PluginCommands::Install(install_args) => install_plugin(install_args),
        PluginCommands::Remove(remove_args) => remove_plugin(remove_args),
        PluginCommands::Update(update_args) => update_plugins(update_args),
        PluginCommands::Search(search_args) => search_plugins(search_args),
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

    // Format plugins
    println!("Format plugins:");
    let formats: Vec<_> = registry.formats().collect();
    if formats.is_empty() {
        println!("  (none installed)");
    } else {
        for plugin in formats {
            let caps = &plugin.capabilities;
            let mut features = Vec::new();
            if caps.load_model.unwrap_or(false) {
                features.push("load_model");
            }
            if caps.save_model.unwrap_or(false) {
                features.push("save_model");
            }
            if caps.load_tensor.unwrap_or(false) {
                features.push("load_tensor");
            }
            if caps.save_tensor.unwrap_or(false) {
                features.push("save_tensor");
            }
            let features_str = if features.is_empty() {
                String::new()
            } else {
                format!("[{}]", features.join(", "))
            };

            let extensions = if caps.extensions.is_empty() {
                String::new()
            } else {
                caps.extensions.join(", ")
            };

            println!(
                "  {:<20} {:<10} {:<40} {:<15} {}",
                plugin.name, plugin.version, features_str, extensions, plugin.source
            );
        }
    }

    Ok(())
}

fn install_plugin(args: InstallArgs) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = &args.path {
        install_from_path(path, args.debug, args.force)
    } else if let Some(git) = &args.git {
        println!("Installing from git: {}", git);
        println!("hodu plugin install --git: not yet implemented");
        Ok(())
    } else if let Some(name) = &args.name {
        println!("Installing from crates.io: {}", name);
        println!("hodu plugin install <name>: not yet implemented");
        Ok(())
    } else if let Some(bundle) = &args.bundle {
        println!("Installing bundle: {}", bundle);
        println!("hodu plugin install --bundle: not yet implemented");
        Ok(())
    } else {
        Err("No plugin specified. Use --path, --git, --bundle, or provide a plugin name.".into())
    }
}

fn install_from_path(path: &Path, debug: bool, force: bool) -> Result<(), Box<dyn std::error::Error>> {
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

    // Build the plugin with cargo, specifying the package
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

    // Find the built library
    let profile = if debug { "debug" } else { "release" };

    // Look for dylib/so file
    let lib_ext = if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    };

    // Convert package name to library name (replace - with _)
    let lib_name = package_name.replace('-', "_");

    // Try multiple possible target directories:
    // 1. Direct target dir (standalone crate)
    // 2. Parent workspace target dir (workspace member)
    let possible_target_dirs = vec![
        path.join("target").join(profile),
        path.parent()
            .map(|p| p.join("target").join(profile))
            .unwrap_or_default(),
    ];

    let mut lib_path = None;
    for target_dir in &possible_target_dirs {
        if !target_dir.exists() {
            continue;
        }

        // Look for library matching the package name
        let expected_lib = format!("lib{}.{}", lib_name, lib_ext);
        let candidate = target_dir.join(&expected_lib);
        if candidate.exists() {
            lib_path = Some(candidate);
            break;
        }

        // Fallback: look for any matching library
        let lib_files: Vec<_> = std::fs::read_dir(target_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                name.starts_with("lib") && name.contains(&lib_name) && name.ends_with(&format!(".{}", lib_ext))
            })
            .collect();

        if !lib_files.is_empty() {
            lib_path = Some(lib_files[0].path());
            break;
        }
    }

    let lib_path = lib_path.ok_or_else(|| {
        format!(
            "No library (.{}) found for package '{}'. Checked: {:?}",
            lib_ext, package_name, possible_target_dirs
        )
    })?;
    println!("Found library: {}", lib_path.display());

    // Detect plugin type and get metadata
    let detected = detect_plugin_type(&lib_path)?;

    let (name, version, sdk_version, plugin_type, capabilities) = match &detected {
        DetectedPluginType::Backend {
            name,
            version,
            sdk_version,
        } => {
            println!("Detected backend plugin: {} v{}", name, version);

            // Load the plugin to get capabilities
            let loaded = LoadedBackendPlugin::load(&lib_path)?;
            let plugin = loaded.plugin();
            let caps = plugin.capabilities();
            let devices: Vec<String> = plugin.supported_devices().iter().map(|d| format!("{:?}", d)).collect();
            let targets: Vec<String> = plugin.supported_targets().iter().map(|t| t.triple.clone()).collect();

            (
                name.clone(),
                version.clone(),
                sdk_version.clone(),
                PluginType::Backend,
                PluginCapabilities::backend(caps.has_runner(), caps.has_builder(), devices, targets),
            )
        },
        DetectedPluginType::Format {
            name,
            version,
            sdk_version,
        } => {
            println!("Detected format plugin: {} v{}", name, version);

            // Load the plugin to get capabilities
            let loaded = LoadedFormatPlugin::load(&lib_path)?;
            let plugin = loaded.plugin();
            let caps = plugin.capabilities();
            let extensions: Vec<String> = plugin
                .supported_extensions()
                .iter()
                .map(|e| format!(".{}", e))
                .collect();

            (
                name.clone(),
                version.clone(),
                sdk_version.clone(),
                PluginType::Format,
                PluginCapabilities::format(
                    caps.has_load_model(),
                    caps.has_save_model(),
                    caps.has_load_tensor(),
                    caps.has_save_tensor(),
                    extensions,
                ),
            )
        },
    };

    // Check SDK version compatibility
    // Policy: host_major == plugin_major && host_minor >= plugin_minor
    // e.g., host=0.4.0, plugin=0.3.0 → OK (host is newer)
    // e.g., host=0.3.0, plugin=0.4.0 → Error (plugin requires newer SDK)
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

    // Copy library to plugins directory
    let plugins_dir = get_plugins_dir()?;
    let lib_filename = lib_path.file_name().unwrap().to_string_lossy().to_string();
    let dest_path = plugins_dir.join(&lib_filename);

    println!("Copying to: {}", dest_path.display());
    std::fs::copy(&lib_path, &dest_path)?;

    // Create registry entry
    let entry = PluginEntry {
        name: name.clone(),
        version,
        plugin_type,
        capabilities,
        library: lib_filename,
        source: PluginSource::Local(path.to_string_lossy().to_string()),
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
    // Simple ISO 8601 timestamp without chrono dependency
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
        // Try with hodu-backend- or hodu-format- prefix
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
    let library = plugin.library.clone();
    let name = plugin.name.clone();
    let version = plugin.version.clone();

    // Delete the library file
    let plugins_dir = get_plugins_dir()?;
    let lib_path = plugins_dir.join(&library);

    if lib_path.exists() {
        std::fs::remove_file(&lib_path)?;
        println!("Removed library: {}", lib_path.display());
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

    let formats: Vec<_> = registry.formats().collect();
    if !formats.is_empty() {
        result.push_str("  Format: ");
        result.push_str(&formats.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", "));
        result.push('\n');
    }

    if result.is_empty() {
        result.push_str("  (none installed)\n");
    }

    result
}

fn update_plugins(args: UpdateArgs) -> Result<(), Box<dyn std::error::Error>> {
    // TODO: Implement plugin update
    if let Some(name) = &args.name {
        println!("Updating plugin: {}", name);
    } else {
        println!("Updating all plugins");
    }
    println!("hodu plugin update: not yet implemented");

    Ok(())
}

fn search_plugins(args: SearchArgs) -> Result<(), Box<dyn std::error::Error>> {
    // TODO: Implement crates.io search
    println!("Searching for: {}", args.query);
    if let Some(t) = &args.plugin_type {
        println!("Type filter: {}", t);
    }
    println!("hodu plugin search: not yet implemented");

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
    // Simple TOML parsing for [package] name = "..."
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
            // name = "package-name"
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
    use hodu_cli_plugin_sdk::{
        build_rs_template, cargo_toml_template, info_toml_backend_template, info_toml_format_template,
        lib_rs_backend_template, lib_rs_format_template,
    };

    let plugin_type = args.plugin_type.to_lowercase();
    if plugin_type != "backend" && plugin_type != "format" {
        return Err(format!(
            "Invalid plugin type: '{}'. Use 'backend' or 'format'.",
            args.plugin_type
        )
        .into());
    }

    // Determine output directory
    let output_dir = args.output.unwrap_or_else(|| std::env::current_dir().unwrap());
    let project_dir = output_dir.join(&args.name);

    if project_dir.exists() {
        return Err(format!("Directory already exists: {}", project_dir.display()).into());
    }

    println!("Creating {} plugin: {}", plugin_type, args.name);

    // Create project structure
    std::fs::create_dir_all(&project_dir)?;
    std::fs::create_dir_all(project_dir.join("src"))?;

    // Generate struct name from plugin name (e.g., hodu-backend-cpu -> HoduBackendCpu)
    let struct_name: String = args
        .name
        .split(['-', '_'])
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars).collect(),
            }
        })
        .collect();

    // Write Cargo.toml
    let cargo_toml = cargo_toml_template(&args.name);
    std::fs::write(project_dir.join("Cargo.toml"), cargo_toml)?;

    // Write info.toml
    let info_toml = match plugin_type.as_str() {
        "backend" => info_toml_backend_template(&args.name, "0.1.0"),
        "format" => info_toml_format_template(&args.name, "0.1.0"),
        _ => unreachable!(),
    };
    std::fs::write(project_dir.join("info.toml"), info_toml)?;

    // Write build.rs
    let build_rs = build_rs_template();
    std::fs::write(project_dir.join("build.rs"), build_rs)?;

    // Write src/lib.rs
    let lib_rs = match plugin_type.as_str() {
        "backend" => lib_rs_backend_template(&struct_name),
        "format" => lib_rs_format_template(&struct_name),
        _ => unreachable!(),
    };
    std::fs::write(project_dir.join("src").join("lib.rs"), lib_rs)?;

    println!("Created plugin project at: {}", project_dir.display());
    println!();
    println!("Next steps:");
    println!("  1. cd {}", args.name);
    println!("  2. Edit info.toml to configure your plugin");
    println!("  3. Implement the plugin in src/lib.rs");
    println!("  4. Install with: hodu plugin install --path .");

    Ok(())
}
