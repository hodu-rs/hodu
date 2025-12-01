//! Proc macros for Hodu CLI plugin development
//!
//! Provides derive macros for plugin development:
//! - `#[derive(BackendPlugin)]`: For backend plugins (runner/builder)
//! - `#[derive(FormatPlugin)]`: For format plugins (load/save model/tensor)
//!
//! These macros read plugin metadata from environment variables set by build.rs
//! and automatically implement the plugin traits.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

/// SDK version - evaluated at proc-macro compile time
const SDK_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Derive macro for Backend plugins
///
/// This macro automatically implements `BackendPlugin` trait using metadata
/// from environment variables set by `hodu_cli_plugin_sdk::build()`.
///
/// The user only needs to implement a `run()` method on their struct:
///
/// ```ignore
/// use hodu_cli_plugin_sdk::*;
///
/// #[derive(Default, BackendPlugin)]
/// pub struct MyBackend;
///
/// impl MyBackend {
///     pub fn run(
///         &self,
///         snapshot: &Snapshot,
///         device: Device,
///         inputs: &[(&str, TensorData)],
///     ) -> PluginResult<HashMap<String, TensorData>> {
///         // Implementation
///     }
/// }
/// ```
#[proc_macro_derive(BackendPlugin)]
pub fn derive_backend_plugin(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let sdk_version = SDK_VERSION;

    let expanded = quote! {
        impl ::hodu_cli_plugin_sdk::BackendPlugin for #name {
            fn name(&self) -> &str {
                env!("HODU_PLUGIN_NAME")
            }

            fn version(&self) -> &str {
                env!("HODU_PLUGIN_VERSION")
            }

            fn capabilities(&self) -> ::hodu_cli_plugin_sdk::BackendCapabilities {
                let mut caps = ::hodu_cli_plugin_sdk::BackendCapabilities::NONE;
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_BACKEND_RUNNER")) {
                    caps = caps | ::hodu_cli_plugin_sdk::BackendCapabilities::RUNNER;
                }
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_BACKEND_BUILDER")) {
                    caps = caps | ::hodu_cli_plugin_sdk::BackendCapabilities::BUILDER;
                }
                caps
            }

            fn supported_devices(&self) -> Vec<::hodu_cli_plugin_sdk::Device> {
                ::hodu_cli_plugin_sdk::__parse_devices(env!("HODU_BACKEND_DEVICES"))
            }

            fn run(
                &self,
                snapshot: &::hodu_cli_plugin_sdk::Snapshot,
                device: ::hodu_cli_plugin_sdk::Device,
                inputs: &[(&str, ::hodu_cli_plugin_sdk::TensorData)],
            ) -> ::hodu_cli_plugin_sdk::PluginResult<::std::collections::HashMap<String, ::hodu_cli_plugin_sdk::TensorData>> {
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_BACKEND_RUNNER")) {
                    #name::run(self, snapshot, device, inputs)
                } else {
                    Err(::hodu_cli_plugin_sdk::PluginError::NotSupported("runner".into()))
                }
            }

            fn supported_targets(&self) -> Vec<::hodu_cli_plugin_sdk::BuildTarget> {
                ::hodu_cli_plugin_sdk::__parse_targets(env!("HODU_BACKEND_TARGETS"))
            }

            fn supported_formats(&self, target: &::hodu_cli_plugin_sdk::BuildTarget) -> Vec<::hodu_cli_plugin_sdk::BuildFormat> {
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_BACKEND_BUILDER")) {
                    #name::supported_formats(self, target)
                } else {
                    vec![]
                }
            }

            fn build(
                &self,
                snapshot: &::hodu_cli_plugin_sdk::Snapshot,
                target: &::hodu_cli_plugin_sdk::BuildTarget,
                format: ::hodu_cli_plugin_sdk::BuildFormat,
                output: &::std::path::Path,
            ) -> ::hodu_cli_plugin_sdk::PluginResult<()> {
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_BACKEND_BUILDER")) {
                    #name::build(self, snapshot, target, format, output)
                } else {
                    Err(::hodu_cli_plugin_sdk::PluginError::NotSupported("builder".into()))
                }
            }
        }

        // Export FFI functions for backend plugin
        const _: () = {
            #[no_mangle]
            pub extern "C" fn hodu_backend_plugin_create() -> *mut ::hodu_cli_plugin_sdk::BackendPluginHandle {
                let boxed: Box<dyn ::hodu_cli_plugin_sdk::BackendPlugin> = Box::new(<#name>::default());
                ::hodu_cli_plugin_sdk::BackendPluginHandle::from_boxed(boxed)
            }

            #[no_mangle]
            pub unsafe extern "C" fn hodu_backend_plugin_destroy(ptr: *mut ::hodu_cli_plugin_sdk::BackendPluginHandle) {
                if !ptr.is_null() {
                    drop(::hodu_cli_plugin_sdk::BackendPluginHandle::into_boxed(ptr));
                }
            }

            #[no_mangle]
            pub extern "C" fn hodu_backend_plugin_metadata() -> ::hodu_cli_plugin_sdk::BackendPluginMetadata {
                ::hodu_cli_plugin_sdk::BackendPluginMetadata {
                    ffi_version: ::hodu_cli_plugin_sdk::FFI_PROTOCOL_VERSION,
                    sdk_version: concat!(#sdk_version, "\0").as_ptr().cast(),
                    name: concat!(env!("HODU_PLUGIN_NAME"), "\0").as_ptr().cast(),
                    version: concat!(env!("HODU_PLUGIN_VERSION"), "\0").as_ptr().cast(),
                    _reserved: [0; 4],
                }
            }
        };
    };

    TokenStream::from(expanded)
}

/// Derive macro for Format plugins
///
/// This macro automatically implements `FormatPlugin` trait using metadata
/// from environment variables set by `hodu_cli_plugin_sdk::build()`.
///
/// The user only needs to implement the methods enabled in info.toml:
///
/// ```ignore
/// use hodu_cli_plugin_sdk::*;
///
/// #[derive(Default, FormatPlugin)]
/// pub struct MyFormat;
///
/// impl MyFormat {
///     // Required when load_model = true in info.toml
///     pub fn load_model(&self, path: &Path) -> PluginResult<Snapshot> {
///         // Implementation
///     }
///
///     pub fn load_model_from_bytes(&self, data: &[u8]) -> PluginResult<Snapshot> {
///         // Implementation
///     }
/// }
/// ```
#[proc_macro_derive(FormatPlugin)]
pub fn derive_format_plugin(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let sdk_version = SDK_VERSION;

    let expanded = quote! {
        impl ::hodu_cli_plugin_sdk::FormatPlugin for #name {
            fn name(&self) -> &str {
                env!("HODU_PLUGIN_NAME")
            }

            fn version(&self) -> &str {
                env!("HODU_PLUGIN_VERSION")
            }

            fn capabilities(&self) -> ::hodu_cli_plugin_sdk::FormatCapabilities {
                let mut caps = ::hodu_cli_plugin_sdk::FormatCapabilities::NONE;
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_FORMAT_LOAD_MODEL")) {
                    caps = caps | ::hodu_cli_plugin_sdk::FormatCapabilities::LOAD_MODEL;
                }
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_FORMAT_SAVE_MODEL")) {
                    caps = caps | ::hodu_cli_plugin_sdk::FormatCapabilities::SAVE_MODEL;
                }
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_FORMAT_LOAD_TENSOR")) {
                    caps = caps | ::hodu_cli_plugin_sdk::FormatCapabilities::LOAD_TENSOR;
                }
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_FORMAT_SAVE_TENSOR")) {
                    caps = caps | ::hodu_cli_plugin_sdk::FormatCapabilities::SAVE_TENSOR;
                }
                caps
            }

            fn supported_extensions(&self) -> Vec<&str> {
                ::hodu_cli_plugin_sdk::__parse_extensions(env!("HODU_FORMAT_EXTENSIONS"))
            }

            fn load_model(&self, path: &::std::path::Path) -> ::hodu_cli_plugin_sdk::PluginResult<::hodu_cli_plugin_sdk::Snapshot> {
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_FORMAT_LOAD_MODEL")) {
                    #name::load_model(self, path)
                } else {
                    Err(::hodu_cli_plugin_sdk::PluginError::NotSupported("load_model".into()))
                }
            }

            fn load_model_from_bytes(&self, data: &[u8]) -> ::hodu_cli_plugin_sdk::PluginResult<::hodu_cli_plugin_sdk::Snapshot> {
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_FORMAT_LOAD_MODEL")) {
                    #name::load_model_from_bytes(self, data)
                } else {
                    Err(::hodu_cli_plugin_sdk::PluginError::NotSupported("load_model".into()))
                }
            }

            fn save_model(&self, snapshot: &::hodu_cli_plugin_sdk::Snapshot, path: &::std::path::Path) -> ::hodu_cli_plugin_sdk::PluginResult<()> {
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_FORMAT_SAVE_MODEL")) {
                    #name::save_model(self, snapshot, path)
                } else {
                    Err(::hodu_cli_plugin_sdk::PluginError::NotSupported("save_model".into()))
                }
            }

            fn save_model_to_bytes(&self, snapshot: &::hodu_cli_plugin_sdk::Snapshot) -> ::hodu_cli_plugin_sdk::PluginResult<Vec<u8>> {
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_FORMAT_SAVE_MODEL")) {
                    #name::save_model_to_bytes(self, snapshot)
                } else {
                    Err(::hodu_cli_plugin_sdk::PluginError::NotSupported("save_model".into()))
                }
            }

            fn load_tensor(&self, path: &::std::path::Path) -> ::hodu_cli_plugin_sdk::PluginResult<::hodu_cli_plugin_sdk::TensorData> {
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_FORMAT_LOAD_TENSOR")) {
                    #name::load_tensor(self, path)
                } else {
                    Err(::hodu_cli_plugin_sdk::PluginError::NotSupported("load_tensor".into()))
                }
            }

            fn load_tensor_from_bytes(&self, data: &[u8]) -> ::hodu_cli_plugin_sdk::PluginResult<::hodu_cli_plugin_sdk::TensorData> {
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_FORMAT_LOAD_TENSOR")) {
                    #name::load_tensor_from_bytes(self, data)
                } else {
                    Err(::hodu_cli_plugin_sdk::PluginError::NotSupported("load_tensor".into()))
                }
            }

            fn save_tensor(&self, tensor: &::hodu_cli_plugin_sdk::TensorData, path: &::std::path::Path) -> ::hodu_cli_plugin_sdk::PluginResult<()> {
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_FORMAT_SAVE_TENSOR")) {
                    #name::save_tensor(self, tensor, path)
                } else {
                    Err(::hodu_cli_plugin_sdk::PluginError::NotSupported("save_tensor".into()))
                }
            }

            fn save_tensor_to_bytes(&self, tensor: &::hodu_cli_plugin_sdk::TensorData) -> ::hodu_cli_plugin_sdk::PluginResult<Vec<u8>> {
                if ::hodu_cli_plugin_sdk::__parse_bool(env!("HODU_FORMAT_SAVE_TENSOR")) {
                    #name::save_tensor_to_bytes(self, tensor)
                } else {
                    Err(::hodu_cli_plugin_sdk::PluginError::NotSupported("save_tensor".into()))
                }
            }
        }

        // Export FFI functions for format plugin
        const _: () = {
            #[no_mangle]
            pub extern "C" fn hodu_format_plugin_create() -> *mut ::hodu_cli_plugin_sdk::FormatPluginHandle {
                let boxed: Box<dyn ::hodu_cli_plugin_sdk::FormatPlugin> = Box::new(<#name>::default());
                ::hodu_cli_plugin_sdk::FormatPluginHandle::from_boxed(boxed)
            }

            #[no_mangle]
            pub unsafe extern "C" fn hodu_format_plugin_destroy(ptr: *mut ::hodu_cli_plugin_sdk::FormatPluginHandle) {
                if !ptr.is_null() {
                    drop(::hodu_cli_plugin_sdk::FormatPluginHandle::into_boxed(ptr));
                }
            }

            #[no_mangle]
            pub extern "C" fn hodu_format_plugin_metadata() -> ::hodu_cli_plugin_sdk::FormatPluginMetadata {
                ::hodu_cli_plugin_sdk::FormatPluginMetadata {
                    ffi_version: ::hodu_cli_plugin_sdk::FFI_PROTOCOL_VERSION,
                    sdk_version: concat!(#sdk_version, "\0").as_ptr().cast(),
                    name: concat!(env!("HODU_PLUGIN_NAME"), "\0").as_ptr().cast(),
                    version: concat!(env!("HODU_PLUGIN_VERSION"), "\0").as_ptr().cast(),
                    _reserved: [0; 4],
                }
            }
        };
    };

    TokenStream::from(expanded)
}
