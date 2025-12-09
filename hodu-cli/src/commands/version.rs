use crate::plugins::PluginRegistry;

pub fn execute() -> Result<(), Box<dyn std::error::Error>> {
    println!("hodu {}", env!("CARGO_PKG_VERSION"));
    println!("hodu-plugin {}", hodu_plugin::PLUGIN_VERSION);

    #[cfg(target_os = "macos")]
    let platform = format!("{}-apple-darwin", std::env::consts::ARCH);
    #[cfg(target_os = "linux")]
    let platform = format!("{}-unknown-linux-gnu", std::env::consts::ARCH);
    #[cfg(target_os = "windows")]
    let platform = format!("{}-pc-windows-msvc", std::env::consts::ARCH);
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    let platform = format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS);

    println!("Platform: {}", platform);

    // List installed plugins
    println!();
    println!("Installed plugins:");

    match PluginRegistry::default_path() {
        Ok(registry_path) => match PluginRegistry::load(&registry_path) {
            Ok(registry) => {
                let backends: Vec<_> = registry.backends().collect();
                let model_formats: Vec<_> = registry.model_formats().collect();
                let tensor_formats: Vec<_> = registry.tensor_formats().collect();

                if backends.is_empty() && model_formats.is_empty() && tensor_formats.is_empty() {
                    println!("  (none)");
                } else {
                    for plugin in backends {
                        println!("  {} {} [backend]", plugin.name, plugin.version);
                    }
                    for plugin in model_formats {
                        println!("  {} {} [model_format]", plugin.name, plugin.version);
                    }
                    for plugin in tensor_formats {
                        println!("  {} {} [tensor_format]", plugin.name, plugin.version);
                    }
                }
            },
            Err(_) => println!("  (none)"),
        },
        Err(_) => println!("  (none)"),
    }

    Ok(())
}
