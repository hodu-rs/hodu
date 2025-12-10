//! Cargo-style output formatting
//!
//! Provides consistent, colorful terminal output similar to cargo.

use std::io::{self, Write};

/// ANSI color codes
pub mod colors {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const GREEN: &str = "\x1b[32m";
    pub const CYAN: &str = "\x1b[36m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const RED: &str = "\x1b[31m";
    pub const BOLD_GREEN: &str = "\x1b[1;32m";
    pub const BOLD_CYAN: &str = "\x1b[1;36m";
    pub const BOLD_YELLOW: &str = "\x1b[1;33m";
    pub const BOLD_RED: &str = "\x1b[1;31m";
}

/// Check if terminal supports colors
pub fn supports_color() -> bool {
    std::env::var("NO_COLOR").is_err() && std::env::var("TERM").map(|t| t != "dumb").unwrap_or(true)
}

/// Print a status message in cargo style
/// Format: "   {status} {message}"
fn print_status(status: &str, color: &str, message: &str) {
    if supports_color() {
        eprintln!("{}{:>12}{} {}", color, status, colors::RESET, message);
    } else {
        eprintln!("{:>12} {}", status, message);
    }
}

/// Print "Compiling" status (green)
pub fn compiling(message: &str) {
    print_status("Compiling", colors::BOLD_GREEN, message);
}

/// Print "Building" status (green)
pub fn building(message: &str) {
    print_status("Building", colors::BOLD_GREEN, message);
}

/// Print "Running" status (green)
pub fn running(message: &str) {
    print_status("Running", colors::BOLD_GREEN, message);
}

/// Print "Finished" status (green)
pub fn finished(message: &str) {
    print_status("Finished", colors::BOLD_GREEN, message);
}

/// Print "Installing" status (green)
pub fn installing(message: &str) {
    print_status("Installing", colors::BOLD_GREEN, message);
}

/// Print "Installed" status (green)
pub fn installed(message: &str) {
    print_status("Installed", colors::BOLD_GREEN, message);
}

/// Print "Removing" status (green)
pub fn removing(message: &str) {
    print_status("Removing", colors::BOLD_GREEN, message);
}

/// Print "Removed" status (green)
pub fn removed(message: &str) {
    print_status("Removed", colors::BOLD_GREEN, message);
}

/// Print "Updating" status (green)
pub fn updating(message: &str) {
    print_status("Updating", colors::BOLD_GREEN, message);
}

/// Print "Updated" status (green)
pub fn updated(message: &str) {
    print_status("Updated", colors::BOLD_GREEN, message);
}

/// Print "Cleaning" status (green)
pub fn cleaning(message: &str) {
    print_status("Cleaning", colors::BOLD_GREEN, message);
}

/// Print "Loading" status (cyan)
pub fn loading(message: &str) {
    print_status("Loading", colors::BOLD_CYAN, message);
}

/// Print "Converting" status (cyan)
pub fn converting(message: &str) {
    print_status("Converting", colors::BOLD_CYAN, message);
}

/// Print "Inspecting" status (cyan)
pub fn inspecting(message: &str) {
    print_status("Inspecting", colors::BOLD_CYAN, message);
}

/// Print "Downloading" status (cyan)
pub fn downloading(message: &str) {
    print_status("Downloading", colors::BOLD_CYAN, message);
}

/// Print "Fetching" status (cyan)
pub fn fetching(message: &str) {
    print_status("Fetching", colors::BOLD_CYAN, message);
}

/// Print "Warning" status (yellow)
pub fn warning(message: &str) {
    print_status("Warning", colors::BOLD_YELLOW, message);
}

/// Print "Error" status (red)
pub fn error(message: &str) {
    print_status("Error", colors::BOLD_RED, message);
}

/// Print "Skipping" status (yellow)
pub fn skipping(message: &str) {
    print_status("Skipping", colors::BOLD_YELLOW, message);
}

/// Print "Cached" status (cyan)
pub fn cached(message: &str) {
    print_status("Cached", colors::BOLD_CYAN, message);
}

/// Progress indicator that updates in place
pub struct Progress {
    show_spinner: bool,
}

impl Progress {
    pub fn new(status: &str, message: &str) -> Self {
        if supports_color() {
            eprint!("{}{:>12}{} {}...", colors::BOLD_CYAN, status, colors::RESET, message);
        } else {
            eprint!("{:>12} {}...", status, message);
        }
        let _ = io::stderr().flush();
        Self {
            show_spinner: supports_color(),
        }
    }

    pub fn done(self) {
        eprintln!(" done");
    }

    pub fn fail(self, err_msg: &str) {
        if self.show_spinner {
            eprintln!(" {}failed{}", colors::BOLD_RED, colors::RESET);
            error(err_msg);
        } else {
            eprintln!(" failed");
            eprintln!("       Error: {}", err_msg);
        }
    }
}

/// Format duration in human readable form
pub fn format_duration(secs: f64) -> String {
    if secs < 1.0 {
        format!("{:.0}ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{:.2}s", secs)
    } else {
        let mins = (secs / 60.0).floor();
        let secs = secs % 60.0;
        format!("{:.0}m {:.2}s", mins, secs)
    }
}

/// Format byte size in human readable form
pub fn format_size(bytes: u64) -> String {
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
