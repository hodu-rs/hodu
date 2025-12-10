//! Completions command - generate shell completion scripts

use clap::{Args, CommandFactory, ValueEnum};
use clap_complete::{generate, Shell};
use std::io;

#[derive(Args)]
pub struct CompletionsArgs {
    /// Shell to generate completions for
    #[arg(value_enum)]
    pub shell: ShellType,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum ShellType {
    Bash,
    Zsh,
    Fish,
    PowerShell,
    Elvish,
}

impl From<ShellType> for Shell {
    fn from(shell: ShellType) -> Self {
        match shell {
            ShellType::Bash => Shell::Bash,
            ShellType::Zsh => Shell::Zsh,
            ShellType::Fish => Shell::Fish,
            ShellType::PowerShell => Shell::PowerShell,
            ShellType::Elvish => Shell::Elvish,
        }
    }
}

pub fn execute<C: CommandFactory>(args: CompletionsArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = C::command();
    let shell: Shell = args.shell.into();
    generate(shell, &mut cmd, "hodu", &mut io::stdout());
    Ok(())
}
