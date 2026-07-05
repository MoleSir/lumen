use proc_macro2::TokenStream;
use quote::quote;
use crate::utils;

use super::common;

pub fn generate_struct(ast: &syn::DeriveInput) -> TokenStream {
    let lumen = utils::get_lumen_nn_path();
    let name = &ast.ident;

    // extract generic param
    let validated_generic = match common::validate_and_extract_generic(&ast.generics) {
        Ok(res) => res,
        Err(e) => return e.to_compile_error().into(),
    };

    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let (impl_generics_tokens, module_generic_type) = match validated_generic.as_ref() {
        Some(user_generic_ident) => {
            (quote! { #impl_generics }, quote! { #user_generic_ident })
        }
        None => {
            (quote! { <T: lumen_core::FloatDType> }, quote! { T })
        }
    };

    // module attr
    let mut custom_display_fn_name: Option<syn::Ident> = None;
    let mut custom_set_train_name: Option<syn::Ident> = None;
    
    for attr in &ast.attrs {
        if attr.path().is_ident("module") {
            let _ = attr.parse_nested_meta(|meta| {
                // module display
                if meta.path.is_ident("display") {
                    let value = meta.value()?; 
                    let s: syn::LitStr = value.parse()?; 
                    custom_display_fn_name = Some(syn::Ident::new(&s.value(), s.span()));
                    return Ok(());
                }          
                if meta.path.is_ident("train") {
                    let value = meta.value()?; 
                    let s: syn::LitStr = value.parse()?; 
                    custom_set_train_name = Some(syn::Ident::new(&s.value(), s.span()));
                    return Ok(());
                }          
                      
                Ok(())
            });
        }
    }
    
    // module display
    let extra_display_fn = if let Some(fn_name) = custom_display_fn_name {
        quote! {
            fn extra_display(&self) -> String {
                self.#fn_name()
            }
        }
    } else {
        quote! {}
    };

    let display_body = match validated_generic.as_ref() {
        Some(_) => quote! {
            use #lumen::modules::Module;
            write!(f, "{}", self.display())
        },
        None => quote! {
            use #lumen::modules::Module;
            write!(f, "{}", Module::<f64>::display(&*self))
        },
    };

    // set train
    let set_train_fn = if let Some(fn_name) = custom_set_train_name {
        quote! {
            fn set_train(&mut self, mode: bool) {
                self.#fn_name(mode)
            }
        }
    } else {
        quote! {}
    };

    // generate visit filed
    let mut param_body = quote! {};
    let mut param_mut_body = quote! {};
    let mut state_body = quote! {};
    let mut state_mut_body = quote! {};
    let mut buffer_body = quote! {};
    let mut buffer_mut_body = quote! {};
    let mut module_body = quote! {};
    let mut module_mut_body = quote! {};
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
                param_body.extend(field_code);

                let field_code = quote! {
                    visitor.enter_submodule(#name_str, &mut self.#name);
                    #lumen::modules::Module::visit_param_mut(&mut self.#name, visitor)?;
                    visitor.exit_submodule(#name_str, &mut self.#name);
                };
                param_mut_body.extend(field_code);

                let field_code = quote! {
                    visitor.enter_submodule(#name_str, &self.#name);
                    #lumen::modules::Module::visit_buffer(&self.#name, visitor)?;
                    visitor.exit_submodule(#name_str, &self.#name);
                };
                buffer_body.extend(field_code);

                let field_code = quote! {
                    visitor.enter_submodule(#name_str, &mut self.#name);
                    #lumen::modules::Module::visit_buffer_mut(&mut self.#name, visitor)?;
                    visitor.exit_submodule(#name_str, &mut self.#name);
                };
                buffer_mut_body.extend(field_code);

                let field_code = quote! {
                    visitor.enter_submodule(#name_str, &self.#name);
                    #lumen::modules::Module::visit_state(&self.#name, visitor)?;
                    visitor.exit_submodule(#name_str, &self.#name);
                };
                state_body.extend(field_code);

                let field_code = quote! {
                    visitor.enter_submodule(#name_str, &mut self.#name);
                    #lumen::modules::Module::visit_state_mut(&mut self.#name, visitor)?;
                    visitor.exit_submodule(#name_str, &mut self.#name);
                };
                state_mut_body.extend(field_code);
                
                let field_code = quote! {
                    // handle submodule 
                    visitor.enter_submodule(#name_str, &self.#name)?;
                    // submodule recv
                    #lumen::modules::Module::visit_module(&self.#name, visitor)?;
                    // exit submodule 
                    visitor.exit_submodule(#name_str, &self.#name)?;
                };
                module_body.extend(field_code);
                
                let field_code = quote! {
                    // handle submodule 
                    visitor.enter_submodule(#name_str, &mut self.#name)?;
                    // submodule recv
                    #lumen::modules::Module::visit_module_mut(&mut self.#name, visitor)?;
                    // exit submodule 
                    visitor.exit_submodule(#name_str, &mut self.#name)?;
                };
                module_mut_body.extend(field_code);
            }
        }
        syn::Data::Enum(_) => panic!("Only struct can be derived"),
        syn::Data::Union(_) => panic!("Only struct can be derived"),
    };

    // codegen
    let codegen = quote! {
        impl #impl_generics_tokens #lumen::modules::Module<#module_generic_type> for #name #ty_generics #where_clause {
            
            fn visit_param<Visitor: #lumen::modules::TensorVisitor<#module_generic_type>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                #param_body
                Ok(())
            }

            fn visit_param_mut<Visitor: #lumen::modules::TensorVisitorMut<#module_generic_type>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                #param_mut_body
                Ok(())
            }

            fn visit_buffer<Visitor: #lumen::modules::TensorVisitor<#module_generic_type>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                #buffer_body
                Ok(())
            }

            fn visit_buffer_mut<Visitor: #lumen::modules::TensorVisitorMut<#module_generic_type>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                #buffer_mut_body
                Ok(())
            }

            fn visit_state<Visitor: #lumen::modules::TensorVisitor<#module_generic_type>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                #state_body
                Ok(())
            }

            fn visit_state_mut<Visitor: #lumen::modules::TensorVisitorMut<#module_generic_type>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                #state_mut_body
                Ok(())
            }

            fn visit_module<Visitor: #lumen::modules::ModuleVisitor<#module_generic_type>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                visitor.visit_module(self)?;
                #module_body
                visitor.visit_module_end(self)?;
                Ok(())
            }

            fn visit_module_mut<Visitor: #lumen::modules::ModuleVisitorMut<#module_generic_type>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                visitor.visit_module_mut(self)?;
                #module_mut_body
                visitor.visit_module_mut_end(self)?;
                Ok(())
            }

            #extra_display_fn
            #set_train_fn
        }

        impl #impl_generics std::fmt::Display for #name #ty_generics #where_clause {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                #display_body
            }
        }
    };

    // panic!("{}", codegen);

    codegen
}
