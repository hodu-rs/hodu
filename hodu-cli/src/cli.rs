pub mod commands;
pub mod plugin;

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

    /// Inspect a model file
    Inspect(commands::inspect::InspectArgs),

    /// Manage plugins
    Plugin(commands::plugin::PluginArgs),

    /// Show version information
    Version,
}
