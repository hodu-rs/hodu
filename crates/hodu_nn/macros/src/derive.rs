use crate::manifest::Manifest;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

pub fn derive_module_impl(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let manifest = Manifest::default();
    let hodu_core_path = manifest.get_path("hodu_core");
    let hodu_nn_path = manifest.get_path("hodu_nn");

    let num_inputs = ast
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("module"))
        .and_then(|attr| {
            attr.parse_args::<syn::ExprAssign>().ok().and_then(|expr| {
                if let syn::Expr::Path(left) = *expr.left {
                    if left.path.is_ident("inputs") {
                        if let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Int(n), ..
                        }) = *expr.right
                        {
                            n.base10_parse::<usize>().ok()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
        })
        .unwrap_or(1);

    let input_type = if num_inputs == 1 {
        quote!(&#hodu_core_path::tensor::Tensor)
    } else {
        let tensor_refs = std::iter::repeat_n(quote!(&#hodu_core_path::tensor::Tensor), num_inputs).collect::<Vec<_>>();
        quote!((#(#tensor_refs),*))
    };

    let layer_impl = quote! {
        impl #impl_generics #hodu_nn_path::module::Module<#input_type>
            for #name #ty_generics #where_clause
        {
            fn forward(&self, input: #input_type)
                -> #hodu_core_path::error::HoduResult<#hodu_core_path::tensor::Tensor> {
                self.forward(input)
            }
            fn parameters(&mut self) -> Vec<&mut #hodu_core_path::tensor::Tensor> {
                self.parameters()
            }
        }
    };

    let expanded = quote! {
        #layer_impl
    };

    TokenStream::from(expanded)
}

pub fn derive_optimizer_impl(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let manifest = Manifest::default();
    let hodu_core_path = manifest.get_path("hodu_core");
    let hodu_nn_path = manifest.get_path("hodu_nn");

    let expanded = quote! {
        impl #impl_generics #hodu_nn_path::optimizer::Optimizer for #name #ty_generics #where_clause {
            fn step(&mut self, parameters: &mut [&mut #hodu_core_path::tensor::Tensor])
                -> #hodu_core_path::error::HoduResult<()> {
                self.step(parameters)
            }
            fn zero_grad(&mut self, parameters: &mut [&mut #hodu_core_path::tensor::Tensor])
                -> #hodu_core_path::error::HoduResult<()> {
                self.zero_grad(parameters)
            }
            fn set_learning_rate(&mut self, learning_rate: impl Into<#hodu_core_path::scalar::Scalar>) {
                self.set_learning_rate(learning_rate)
            }
        }
    };

    TokenStream::from(expanded)
}
