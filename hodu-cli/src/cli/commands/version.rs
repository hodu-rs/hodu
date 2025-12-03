pub fn execute() -> Result<(), Box<dyn std::error::Error>> {
    println!("hodu {}", env!("CARGO_PKG_VERSION"));
    println!("hodu-plugin-sdk {}", hodu_plugin_sdk::SDK_VERSION);

    #[cfg(target_os = "macos")]
    let platform = format!("{}-apple-darwin", std::env::consts::ARCH);
    #[cfg(target_os = "linux")]
    let platform = format!("{}-unknown-linux-gnu", std::env::consts::ARCH);
    #[cfg(target_os = "windows")]
    let platform = format!("{}-pc-windows-msvc", std::env::consts::ARCH);
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    let platform = format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS);

    println!("Platform: {}", platform);

    // TODO: List installed plugins
    println!();
    println!("Installed plugins:");
    println!("  (none)");

    Ok(())
}
