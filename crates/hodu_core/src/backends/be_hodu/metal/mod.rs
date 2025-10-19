#[cfg(all(feature = "metal", not(feature = "std")))]
compile_error!(
    "Metal backend requires 'std' feature to be enabled. Please use '--features \"metal,std\"' instead of '--features metal'"
);

#[cfg(feature = "metal")]
pub mod device;
#[cfg(not(feature = "metal"))]
mod dummy;
#[cfg(feature = "metal")]
pub mod storage;

#[cfg(not(feature = "metal"))]
pub use dummy::*;
