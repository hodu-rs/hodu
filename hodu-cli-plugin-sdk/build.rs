fn main() {
    println!("cargo::rerun-if-changed=Cargo.toml");
    println!("cargo::rustc-env=HOST_TARGET={}", std::env::var("TARGET").unwrap());
    println!("cargo::rustc-env=HODU_PLUGIN_SDK_VERSION={}", env!("CARGO_PKG_VERSION"));
}
