/// OS-provided BLAS configuration
pub struct BlasConfig {
    target_os: String,
}

impl BlasConfig {
    pub fn detect() -> Self {
        Self {
            target_os: std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default(),
        }
    }

    #[allow(clippy::single_match)]
    pub fn apply_to_build(&self, build: &mut cc::Build) {
        match self.target_os.as_str() {
            "macos" => self.configure_accelerate(build),
            // Future OS support:
            // "linux" => self.configure_linux_blas(build),
            // "windows" => self.configure_mkl(build),
            _ => {},
        }
    }

    #[allow(clippy::single_match)]
    pub fn setup_linking(&self) {
        match self.target_os.as_str() {
            "macos" => {
                println!("cargo:rustc-link-lib=framework=Accelerate");
            },
            // Future OS support:
            // "linux" => { println!("cargo:rustc-link-lib=blas"); }
            // "windows" => { println!("cargo:rustc-link-lib=mkl_rt"); }
            _ => {},
        }
    }

    fn configure_accelerate(&self, build: &mut cc::Build) {
        build.define("USE_BLAS", None);
        build.define("ACCELERATE_NEW_LAPACK", None);

        // Get SDK path for Accelerate framework headers
        if let Ok(output) = std::process::Command::new("xcrun").args(["--show-sdk-path"]).output() {
            if output.status.success() {
                let sdk_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let accelerate_include = format!(
                    "{}/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers",
                    sdk_path
                );
                build.include(&accelerate_include);
            }
        }
    }

    // Future: Linux BLAS configuration
    // fn configure_linux_blas(&self, build: &mut cc::Build) {
    //     build.define("USE_BLAS", None);
    //     // Try pkg-config to find system BLAS
    // }

    // Future: Windows MKL configuration
    // fn configure_mkl(&self, build: &mut cc::Build) {
    //     build.define("USE_BLAS", None);
    //     build.define("MKL_ILP64", None);
    // }
}
