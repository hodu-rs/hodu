mod cli;

use clap::Parser;
use cli::{Cli, Commands};

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Run(args) => cli::commands::run::execute(args),
        Commands::Build(args) => cli::commands::build::execute(args),
        Commands::Inspect(args) => cli::commands::inspect::execute(args),
        Commands::Plugin(args) => cli::commands::plugin::execute(args),
        Commands::Version => cli::commands::version::execute(),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
