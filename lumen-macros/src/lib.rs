use proc_macro::TokenStream;
mod utils;
mod module;

/// Derive macro for the module.
#[proc_macro_derive(Module, attributes(module))]
pub fn module_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse(input).unwrap();
    module::derive_impl(&input)
}