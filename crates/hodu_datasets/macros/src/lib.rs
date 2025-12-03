extern crate proc_macro;

mod derive;

use proc_macro::TokenStream;

#[proc_macro_derive(Dataset)]
pub fn derive_dataset(input: TokenStream) -> TokenStream {
    derive::derive_dataset_impl(input)
}
