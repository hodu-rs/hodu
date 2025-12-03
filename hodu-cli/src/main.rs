mod commands;
mod plugins;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "hodu")]
#[command(author, version, about = "hodu", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run a model
    Run(commands::run::RunArgs),

    /// Build a model to native artifact
    Build(commands::build::BuildArgs),

    /// Convert models and tensors between formats
    Convert(commands::convert::ConvertArgs),

    /// Inspect a model file
    Inspect(commands::inspect::InspectArgs),

    /// Manage plugins
    Plugin(commands::plugin::PluginArgs),

    /// Show version information
    Version,
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Run(args) => commands::run::execute(args),
        Commands::Build(args) => commands::build::execute(args),
        Commands::Convert(args) => commands::convert::execute(args),
        Commands::Inspect(args) => commands::inspect::execute(args),
        Commands::Plugin(args) => commands::plugin::execute(args),
        Commands::Version => commands::version::execute(),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
