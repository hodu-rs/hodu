#[cfg(feature = "cuda")]
pub mod device;
#[cfg(feature = "cuda")]
pub mod storage;

mod dummy;
#[cfg(not(feature = "cuda"))]
pub use dummy::*;
