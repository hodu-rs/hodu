use crate::{
    error::{HoduError, HoduResult},
    script::Snapshot,
    types::{Device, Runtime},
    utils::valid,
};

/// Validate that runtime supports the target device
pub fn validate_runtime_device(runtime: Runtime, device: Device) -> HoduResult<()> {
    if !runtime.is_supported(device) {
        return Err(HoduError::UnsupportedDeviceForRuntime(device, runtime));
    }
    Ok(())
}

/// Validate snapshot dtypes for target device
pub fn validate_snapshot(snapshot: &Snapshot, device: Device) -> HoduResult<()> {
    // Validate input dtypes for device
    for input in &snapshot.inputs {
        valid::validate_dtype_for_device(input.dtype, device)?;
    }

    // Validate node output dtypes for device
    for _node in &snapshot.nodes {
        // Check if the node's dtype is supported on this device
        // We need to infer dtype from the operation
        // For now, we can check the output_layout if it has dtype info
        // TODO: Add proper dtype inference through the graph
    }

    Ok(())
}
