use hodu_macro_utils::Manifest;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

pub fn derive_dataset_impl(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let manifest = Manifest::default();
    let hodu_core_path = manifest.get_path("hodu_core");
    let hodu_utils_path = manifest.get_path("hodu_utils");

    let expanded = quote! {
        impl #impl_generics #hodu_utils_path::data::dataset::Dataset
            for #name #ty_generics #where_clause
        {
            fn len(&self) -> usize {
                self.len()
            }

            fn get(&self, index: usize)
                -> #hodu_core_path::error::HoduResult<#hodu_utils_path::data::batch::DataItem> {
                self.get(index)
            }
        }
    };

    TokenStream::from(expanded)
}
