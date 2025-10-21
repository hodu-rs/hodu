extern crate proc_macro;

use proc_macro::TokenStream;
use std::{env, fs, path::PathBuf};
use syn::parse_str;
use toml_edit::{DocumentMut, Item};

pub struct Manifest {
    manifest: DocumentMut,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            manifest: env::var_os("CARGO_MANIFEST_DIR")
                .map(PathBuf::from)
                .map(|mut path| {
                    path.push("Cargo.toml");
                    if !path.exists() {
                        panic!("No Cargo manifest found for crate. Expected: {}", path.display());
                    }
                    let manifest = fs::read_to_string(path.clone())
                        .unwrap_or_else(|_| panic!("Unable to read cargo manifest: {}", path.display()));
                    manifest
                        .parse::<DocumentMut>()
                        .unwrap_or_else(|_| panic!("Failed to parse cargo manifest: {}", path.display()))
                })
                .expect("CARGO_MANIFEST_DIR is not defined."),
        }
    }
}

const HODU: &str = "hodu";
const HODU_INTERNAL: &str = "hodu_internal";

impl Manifest {
    pub fn maybe_get_path(&self, name: &str) -> Option<syn::Path> {
        if name == env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME is not defined.") {
            return Some(parse_str("crate").unwrap());
        }

        fn dep_package(dep: &Item) -> Option<&str> {
            if dep.as_str().is_some() {
                None
            } else {
                dep.get("package").map(|name| name.as_str().unwrap())
            }
        }

        let find_in_deps = |deps: &Item| -> Option<syn::Path> {
            let package = if let Some(dep) = deps.get(name) {
                return Some(Self::parse_str(dep_package(dep).unwrap_or(name)));
            } else if let Some(dep) = deps.get(HODU) {
                dep_package(dep).unwrap_or(HODU)
            } else if let Some(dep) = deps.get(HODU_INTERNAL) {
                dep_package(dep).unwrap_or(HODU_INTERNAL)
            } else {
                return None;
            };

            let mut path = Self::parse_str::<syn::Path>(package);
            if let Some(module) = name.strip_prefix("hodu_") {
                path.segments.push(Self::parse_str(module));
            }
            Some(path)
        };

        let deps = self.manifest.get("dependencies");
        let deps_dev = self.manifest.get("dev-dependencies");

        deps.and_then(find_in_deps).or_else(|| deps_dev.and_then(find_in_deps))
    }

    pub fn get_path(&self, name: &str) -> syn::Path {
        let sanitized_name = name.replace('-', "_");

        self.maybe_get_path(&sanitized_name)
            .unwrap_or_else(|| Self::parse_str(&sanitized_name))
    }

    pub fn try_parse_str<T: syn::parse::Parse>(path: &str) -> Option<T> {
        syn::parse(path.parse::<TokenStream>().ok()?).ok()
    }

    pub fn parse_str<T: syn::parse::Parse>(path: &str) -> T {
        Self::try_parse_str(path).unwrap()
    }
}
