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
    let mut body_mut = quote! {};
    let mut submodule_body = quote! {};
    match &ast.data {
        syn::Data::Struct(struct_data) => {
            for field in struct_data.fields.iter() {
                let should_skip = field.attrs.iter().any(|attr| {
                    if !attr.path().is_ident("module") {
                        return false;
                    }

                    let mut found_skip = false;
                    let _ = attr.parse_nested_meta(|meta| {
                        if meta.path.is_ident("skip") {
                            found_skip = true;
                        }
                        Ok(())
                    });

                    found_skip
                });

                if should_skip {
                    continue;
                }
                
                let name = field.ident.clone().unwrap();
                let name_str = name.to_string();
                let field_code = quote! {
                    visitor.enter_submodule(#name_str, &self.#name);
                    #lumen::modules::Module::visit_param(&self.#name, visitor)?;
                    visitor.exit_submodule(#name_str, &self.#name);
                };
                body.extend(field_code);

                let field_code = quote! {
                    visitor.enter_submodule(#name_str, &mut self.#name);
                    #lumen::modules::Module::visit_param_mut(&mut self.#name, visitor)?;
                    visitor.exit_submodule(#name_str, &mut self.#name);
                };
                body_mut.extend(field_code);
                
                let field_code = quote! {
                    // handle submodule 
                    visitor.enter_submodule(#name_str, &self.#name)?;
                    // submodule recv
                    #lumen::modules::Module::visit_module(&self.#name, visitor)?;
                    // exit submodule 
                    visitor.exit_submodule(#name_str, &self.#name)?;
                };
                submodule_body.extend(field_code);
            }
        }
        syn::Data::Enum(_) => panic!("Only struct can be derived"),
        syn::Data::Union(_) => panic!("Only struct can be derived"),
    };

    let codegen = quote! {
        impl #generics_module #lumen::modules::Module<T> for #name #generics_ty #generics_where {
            fn visit_param<Visitor: #lumen::modules::ParamVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                #body
                Ok(())
            }

            fn visit_param_mut<Visitor: #lumen::modules::ParamVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                #body_mut
                Ok(())
            }

            fn visit_module<Visitor: #lumen::modules::ModuleVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                // handle self
                visitor.visit_module(self)?;
                // sub module
                #submodule_body
                visitor.visit_module_end(self)?;
                Ok(())
            }
        }

        impl #generics_module std::fmt::Display for #name #generics_ty #generics_where {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use #lumen::modules::Module;
                write!(f, "{}", self.display())
            }
        }
    };

    codegen
}