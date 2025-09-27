extern crate proc_macro;

mod derive;
pub(crate) mod manifest;

use proc_macro::TokenStream;

#[proc_macro_derive(Module, attributes(module))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    derive::derive_module_impl(input)
}

#[proc_macro_derive(Optimizer)]
pub fn derive_optimizer(input: TokenStream) -> TokenStream {
    derive::derive_optimizer_impl(input)
}
