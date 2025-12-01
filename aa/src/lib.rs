use hodu_plugin_sdk::{
    BackendPlugin, BuildFormat, BuildTarget, Device, PluginResult, Snapshot, TensorData,
};
use std::collections::HashMap;
use std::path::Path;

#[derive(Default, BackendPlugin)]
pub struct Aa;

impl Aa {
    /// Execute the model on the given device
    ///
    /// Required when `runner = true` in info.toml.
    pub fn run(
        &self,
        _snapshot: &Snapshot,
        _device: Device,
        _inputs: &[(&str, TensorData)],
    ) -> PluginResult<HashMap<String, TensorData>> {
        todo!("Implement model execution")
    }

    // === Builder methods (required when `builder = true` in info.toml) ===

    // /// Return supported output formats for a given target
    // pub fn supported_formats(&self, _target: &BuildTarget) -> Vec<BuildFormat> {
    //     vec![BuildFormat::Object, BuildFormat::StaticLib]
    // }

    // /// Build an AOT artifact
    // pub fn build(
    //     &self,
    //     _snapshot: &Snapshot,
    //     _target: &BuildTarget,
    //     _format: BuildFormat,
    //     _output: &Path,
    // ) -> PluginResult<()> {
    //     todo!("Implement AOT compilation")
    // }
}
