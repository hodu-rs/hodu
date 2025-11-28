//! Hodu CLI - Execute and inspect Hodu models

#[path = "cli/info.rs"]
mod info;
#[path = "cli/run.rs"]
mod run;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "hodu")]
#[command(version, about = "Hodu ML framework CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a .hdss model file
    Run {
        /// Path to the .hdss file
        path: PathBuf,

        /// Device to run on (cpu, cuda:0, metal:0)
        #[arg(short, long, default_value = "cpu")]
        device: String,
    },

    /// Show information about a .hdss model file
    Info {
        /// Path to the .hdss file
        path: PathBuf,
    },
}

fn main() {
    let args = Cli::parse();

    let result = match args.command {
        Commands::Run { path, device } => run::execute(path, &device),
        Commands::Info { path } => info::execute(path),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
