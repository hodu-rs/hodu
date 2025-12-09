//! Plugin update logic

use super::install::{fetch_official_registry, install_from_git, install_from_path, install_from_registry};
use crate::output;
use crate::plugins::{PluginRegistry, PluginSource};
use hodu_plugin_sdk::SDK_VERSION;
use std::path::PathBuf;

pub fn update_plugins(name: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let registry_path = PluginRegistry::default_path()?;
    let registry = PluginRegistry::load(&registry_path)?;

    let plugins_to_update: Vec<_> = if let Some(name) = name {
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
        output::updating(&plugin.name);

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
