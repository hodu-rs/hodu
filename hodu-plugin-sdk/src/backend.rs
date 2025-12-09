//! Backend plugin types
//!
//! Types for backend plugins that execute models on various devices.

use serde::{Deserialize, Serialize};
use std::process::Command;

/// Target device for plugin execution
///
/// Using String for extensibility - plugins can define custom devices.
/// Convention: lowercase with `::` separator for device index.
/// Common values: "cpu", "cuda::0", "metal", "vulkan", "webgpu", "rocm::0"
pub type Device = String;

/// Parse device ID from device string (e.g., "cuda::0" -> 0)
pub fn parse_device_id(device: &str) -> Option<usize> {
    device.split("::").nth(1).and_then(|id| id.parse().ok())
}

/// Get device type from device string (e.g., "cuda::0" -> "cuda")
pub fn device_type(device: &str) -> &str {
    device.split("::").next().unwrap_or(device)
}

/// Build target specification for AOT compilation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BuildTarget {
    /// Target triple (e.g., "x86_64-unknown-linux-gnu", "aarch64-apple-darwin")
    pub triple: String,
    /// Target device (e.g., "cpu", "metal", "cuda::0")
    pub device: String,
}

impl BuildTarget {
    /// Create a new build target
    pub fn new(triple: impl Into<String>, device: impl Into<String>) -> Self {
        Self {
            triple: triple.into(),
            device: device.into(),
        }
    }

    /// Create a build target for the current host system
    pub fn host(device: impl Into<String>) -> Self {
        Self::new(current_host_triple(), device)
    }
}

// ============================================================================
// Build Target Capability
// ============================================================================

/// Supported target definition for manifest.json
///
/// # Example manifest.json
/// ```json
/// {
///   "supported_targets": [
///     {
///       "triple": "x86_64-unknown-linux-gnu",
///       "requires": ["clang|gcc"]
///     },
///     {
///       "triple": "aarch64-apple-darwin",
///       "requires": ["clang"],
///       "host_only": ["*-apple-darwin"]
///     }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportedTarget {
    /// Target triple (e.g., "x86_64-unknown-linux-gnu")
    pub triple: String,

    /// Required tools (e.g., ["clang", "nvcc", "xcrun"])
    /// Use "|" for alternatives: "clang|gcc" means clang OR gcc
    #[serde(default)]
    pub requires: Vec<String>,

    /// Host triples that can build this target (glob patterns)
    /// Empty means any host can build (with proper toolchain)
    /// e.g., ["*-apple-darwin"] means only macOS hosts
    #[serde(default)]
    pub host_only: Vec<String>,
}

/// Result of checking if a target can be built
#[derive(Debug, Clone)]
pub struct BuildCapability {
    /// Whether the target can be built from current host
    pub can_build: bool,
    /// Tools that were found and will be used
    pub available_tools: Vec<String>,
    /// Missing tools that need to be installed
    pub missing_tools: Vec<String>,
    /// Why the build is not possible (if can_build is false)
    pub reason: Option<String>,
}

impl BuildCapability {
    /// Create a capability indicating build is available
    pub fn available(tools: Vec<String>) -> Self {
        Self {
            can_build: true,
            available_tools: tools,
            missing_tools: Vec::new(),
            reason: None,
        }
    }

    /// Create a capability indicating build is not available
    pub fn unavailable(missing: Vec<String>, reason: impl Into<String>) -> Self {
        Self {
            can_build: false,
            available_tools: Vec::new(),
            missing_tools: missing,
            reason: Some(reason.into()),
        }
    }
}

// ============================================================================
// Host / Tool Detection Utilities
// ============================================================================

/// Get the current host triple
pub fn current_host_triple() -> &'static str {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    return "x86_64-unknown-linux-gnu";
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    return "aarch64-unknown-linux-gnu";
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    return "x86_64-apple-darwin";
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    return "aarch64-apple-darwin";
    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    return "x86_64-pc-windows-msvc";
    #[cfg(not(any(
        all(target_arch = "x86_64", target_os = "linux"),
        all(target_arch = "aarch64", target_os = "linux"),
        all(target_arch = "x86_64", target_os = "macos"),
        all(target_arch = "aarch64", target_os = "macos"),
        all(target_arch = "x86_64", target_os = "windows"),
    )))]
    return "unknown";
}

/// Check if a tool is available on the system
pub fn is_tool_available(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if host matches a pattern (supports glob * at start/end)
///
/// # Examples
/// - `host_matches_pattern("aarch64-apple-darwin", "*-apple-darwin")` -> true
/// - `host_matches_pattern("x86_64-linux-gnu", "x86_64-*")` -> true
/// - `host_matches_pattern("aarch64-linux-gnu", "*")` -> true
pub fn host_matches_pattern(host: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        return host.ends_with(suffix);
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return host.starts_with(prefix);
    }
    host == pattern
}

/// Plugin manifest (manifest.json)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub sdk_version: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub devices: Vec<String>,
    #[serde(default)]
    pub extensions: Vec<String>,
    #[serde(default)]
    pub dependencies: Vec<String>,
    #[serde(default)]
    pub supported_targets: Vec<SupportedTarget>,
}

impl PluginManifest {
    /// Load manifest from the standard location (next to executable)
    pub fn load() -> Result<Self, String> {
        let exe_path = std::env::current_exe().map_err(|e| format!("Failed to get executable path: {}", e))?;

        let manifest_path = exe_path
            .parent()
            .map(|p| p.join("manifest.json"))
            .ok_or("No parent directory")?;

        Self::load_from(&manifest_path)
    }

    /// Load manifest from a specific path
    pub fn load_from(path: &std::path::Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| format!("Failed to read manifest: {}", e))?;

        serde_json::from_str(&content).map_err(|e| format!("Failed to parse manifest: {}", e))
    }

    /// Check if a target triple is supported and can be built
    pub fn check_target(&self, triple: &str) -> BuildCapability {
        // Find target in manifest
        let target = match self.supported_targets.iter().find(|t| t.triple == triple) {
            Some(t) => t,
            None => {
                return BuildCapability::unavailable(
                    vec![],
                    format!(
                        "Target '{}' is not in supported_targets. Available: {}",
                        triple,
                        self.supported_targets
                            .iter()
                            .map(|t| t.triple.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                );
            },
        };

        check_build_capability(target)
    }

    /// Get list of buildable targets from current host
    pub fn buildable_targets(&self) -> Vec<(&SupportedTarget, BuildCapability)> {
        self.supported_targets
            .iter()
            .map(|t| (t, check_build_capability(t)))
            .collect()
    }

    /// Format supported targets with build status
    pub fn format_targets(&self) -> String {
        let mut result = String::new();
        let host = current_host_triple();

        result.push_str(&format!("Host: {}\n\n", host));

        for target in &self.supported_targets {
            let cap = check_build_capability(target);
            let status = if cap.can_build { "✓" } else { "✗" };
            result.push_str(&format!("  {} {}\n", status, target.triple));
        }

        result
    }
}

/// Check build capability for a supported target
///
/// This checks:
/// 1. If current host is allowed to build this target (host_only)
/// 2. If required tools are available
pub fn check_build_capability(target: &SupportedTarget) -> BuildCapability {
    let host = current_host_triple();

    // Check host restriction
    if !target.host_only.is_empty() {
        let host_allowed = target.host_only.iter().any(|p| host_matches_pattern(host, p));
        if !host_allowed {
            return BuildCapability::unavailable(
                vec![],
                format!(
                    "Host '{}' cannot build target '{}'. Allowed hosts: {}",
                    host,
                    target.triple,
                    target.host_only.join(", ")
                ),
            );
        }
    }

    // Check required tools
    let mut missing = Vec::new();
    let mut available = Vec::new();

    for req in &target.requires {
        // Handle alternatives (e.g., "clang|gcc")
        let alternatives: Vec<&str> = req.split('|').collect();
        let mut found = None;

        for alt in &alternatives {
            if is_tool_available(alt) {
                found = Some(alt.to_string());
                break;
            }
        }

        if let Some(tool) = found {
            available.push(tool);
        } else {
            missing.push(req.clone());
        }
    }

    if !missing.is_empty() {
        return BuildCapability::unavailable(
            missing.clone(),
            format!("Missing required tools: {}", missing.join(", ")),
        );
    }

    BuildCapability::available(available)
}
