fn main() {
    println!("cargo::rerun-if-changed=Cargo.toml");
    println!("cargo::rustc-env=HOST_TARGET={}", std::env::var("TARGET").unwrap());
}
