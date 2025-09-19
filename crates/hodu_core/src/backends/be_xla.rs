#[cfg(not(feature = "std"))]
compile_error!(
    "XLA backend requires 'std' feature to be enabled. Please use '--features \"xla,std\"' instead of '--features xla'"
);

pub mod executor;
