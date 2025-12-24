use proc_macro2::TokenStream;
use quote::quote;
use crate::utils;

pub fn derive_impl(ast: &syn::DeriveInput) -> proc_macro::TokenStream {
    match &ast.data {
        syn::Data::Struct(_) => generate_struct(ast),
        syn::Data::Enum(_) => panic!("Enum modules aren't supported yet."),
        syn::Data::Union(_) => panic!("Union modules aren't supported yet."),
    }
    .into()
}

fn generate_struct(ast: &syn::DeriveInput) -> TokenStream {
    let lumen = utils::get_lumen_nn_path();
    // 
    let name = &ast.ident;
    let (_, generics_ty, generics_where) = ast.generics.split_for_impl();
    let generics_module = ast.generics.clone();

    // generate visit filed
    let mut body = quote! {};
    match &ast.data {
        syn::Data::Struct(struct_data) => {
            for field in struct_data.fields.iter() {
                let name = field.ident.clone().unwrap();
                let name_str = name.to_string();
                let field_code = quote! {
                    visitor.enter_module(#name_str);
                    #lumen::modules::Module::visit(&self.#name, visitor);
                    visitor.exit_module(#name_str);
                };
                body.extend(field_code);
            }
        }
        syn::Data::Enum(_) => panic!("Only struct can be derived"),
        syn::Data::Union(_) => panic!("Only struct can be derived"),
    };

    let codegen = quote! {
        impl #generics_module #lumen::modules::Module<T> for #name #generics_ty #generics_where {
            fn visit<Visitor: #lumen::modules::ModuleVisitor<T>>(&self, visitor: &mut Visitor) {
                #body
            }
        }
    };

    // panic!("{}", codegen);
    
    codegen
}