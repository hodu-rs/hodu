//! Clean command - remove cached build artifacts

use crate::output;
use clap::Args;
use std::path::PathBuf;

#[derive(Args)]
pub struct CleanArgs {
    /// Only show what would be deleted (dry run)
    #[arg(long)]
    pub dry_run: bool,

    /// Clean specific backend cache only
    #[arg(long)]
    pub backend: Option<String>,

    /// Clean all caches including plugin registry (use with caution)
    #[arg(long)]
    pub all: bool,
}

pub fn execute(args: CleanArgs) -> Result<(), Box<dyn std::error::Error>> {
    let hodu_dir = dirs::home_dir()
        .ok_or("Could not determine home directory")?
        .join(".hodu");

    if !hodu_dir.exists() {
        println!("Nothing to clean.");
        return Ok(());
    }

    let cache_dir = hodu_dir.join("cache");

    if args.all {
        // Clean everything
        clean_directory(&hodu_dir, "all hodu data", args.dry_run)?;
    } else if let Some(backend) = &args.backend {
        // Clean specific backend
        let backend_cache = cache_dir.join(backend);
        if backend_cache.exists() {
            clean_directory(&backend_cache, &format!("{} cache", backend), args.dry_run)?;
        } else {
            // Try with prefix
            let prefixed = format!("hodu-backend-{}-plugin", backend);
            let backend_cache = cache_dir.join(&prefixed);
            if backend_cache.exists() {
                clean_directory(&backend_cache, &format!("{} cache", backend), args.dry_run)?;
            } else {
                println!("No cache found for backend '{}'", backend);
            }
        }
    } else {
        // Clean all caches (default)
        if cache_dir.exists() {
            clean_directory(&cache_dir, "build cache", args.dry_run)?;
        } else {
            println!("Nothing to clean.");
        }
    }

    Ok(())
}

fn clean_directory(path: &PathBuf, name: &str, dry_run: bool) -> Result<(), Box<dyn std::error::Error>> {
    let size = dir_size(path)?;
    let size_str = format_size(size);

    if dry_run {
        output::skipping(&format!("{} ({}) - dry run", name, size_str));
    } else {
        output::cleaning(&format!("{} ({})", name, size_str));
        std::fs::remove_dir_all(path)?;
        output::removed(&format!("{}", name));
    }

    Ok(())
}

fn dir_size(path: &PathBuf) -> Result<u64, Box<dyn std::error::Error>> {
    let mut size = 0;
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                size += dir_size(&path)?;
            } else {
                size += entry.metadata()?.len();
            }
        }
    } else {
        size = std::fs::metadata(path)?.len();
    }
    Ok(size)
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}
