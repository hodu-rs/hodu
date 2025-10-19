// https://github.com/huggingface/candle/blob/main/candle-metal-kernels

pub mod dtype;
pub mod error;
pub mod kernel;
pub mod kernels;
pub mod metal;
pub mod source;
pub mod utils;

use metal::MTLResourceOptions;

pub const RESOURCE_OPTIONS: MTLResourceOptions =
    objc2_metal::MTLResourceOptions(MTLResourceOptions::StorageModeShared.bits());
