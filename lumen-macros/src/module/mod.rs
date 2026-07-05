mod eenum;
mod sstruct;
pub use eenum::*;
pub use sstruct::*;
mod common;

pub fn derive_impl(ast: &syn::DeriveInput) -> proc_macro::TokenStream {
    match &ast.data {
        syn::Data::Struct(_) => generate_struct(ast),
        syn::Data::Enum(_) => generate_enum(ast),
        syn::Data::Union(_) => panic!("Union modules aren't supported yet."),
    }
    .into()
}
