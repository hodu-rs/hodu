//! Hodu CLI - Execute and inspect Hodu models

#[path = "cli/compile.rs"]
mod compile;
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

        /// Device to run on (cpu, cuda:0, metal)
        #[arg(short, long, default_value = "cpu")]
        device: String,

        /// Input tensor file (format: name=path.hdt), can be repeated
        #[arg(short, long, action = clap::ArgAction::Append)]
        input: Vec<String>,

        /// Input tensor files, comma-separated (format: name=path.hdt,name=path.json)
        #[arg(short = 'I', long, value_delimiter = ',')]
        inputs: Vec<String>,
    },

    /// Compile a .hdss model to target format
    Compile {
        /// Path to the .hdss file
        path: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Target device (cpu, metal, cuda:0)
        #[arg(short, long, default_value = "metal")]
        device: String,

        /// Output format (msl, air, metallib, ptx, cubin, llvm-ir, object)
        #[arg(short, long, default_value = "metallib")]
        format: String,

        /// Path to compiler plugin (.dylib/.so/.dll)
        #[arg(short, long)]
        plugin: Option<PathBuf>,
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
        Commands::Run {
            path,
            device,
            input,
            inputs,
        } => run::execute(path, &device, input, inputs),
        Commands::Compile {
            path,
            output,
            device,
            format,
            plugin,
        } => compile::execute(path, output, &device, &format, plugin),
        Commands::Info { path } => info::execute(path),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
