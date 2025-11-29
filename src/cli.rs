//! Hodu CLI - Execute and inspect Hodu models

#[path = "cli/common.rs"]
mod common;
#[path = "cli/compile.rs"]
mod compile;
#[path = "cli/info.rs"]
mod info;
#[path = "cli/output.rs"]
mod output;
#[path = "cli/run.rs"]
mod run;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "hodu")]
#[command(version, about = "Hodu ML framework CLI")]
#[command(after_help = "Use 'hodu <COMMAND> --help' for more information about a command.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a .hdss model file
    Run(run::RunArgs),

    /// Compile a .hdss model to target format
    Compile(compile::CompileArgs),

    /// Show information about a .hdss model file
    Info {
        /// Path to the .hdss file
        path: PathBuf,
    },
}

fn main() {
    let args = Cli::parse();

    let result = match args.command {
        Commands::Run(args) => run::execute(args),
        Commands::Compile(args) => compile::execute(args),
        Commands::Info { path } => info::execute(path),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
