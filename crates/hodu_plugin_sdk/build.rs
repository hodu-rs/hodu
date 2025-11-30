fn main() {
    println!("cargo::rustc-env=HOST_TARGET={}", std::env::var("TARGET").unwrap());
}
