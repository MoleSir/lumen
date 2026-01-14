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
    let name = &ast.ident;

    // extract generic param
    let validated_generic = match validate_and_extract_generic(&ast.generics) {
        Ok(res) => res,
        Err(e) => return e.to_compile_error().into(),
    };

    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let (impl_generics_tokens, module_generic_type) = match validated_generic.as_ref() {
        Some(user_generic_ident) => {
            (quote! { #impl_generics }, quote! { #user_generic_ident })
        }
        None => {
            (quote! { <T: FloatDType> }, quote! { T })
        }
    };

    // module attr
    let mut custom_repr_fn_name: Option<syn::Ident> = None;
    for attr in &ast.attrs {
        if attr.path().is_ident("module") {
            let _ = attr.parse_nested_meta(|meta| {
                // module display
                if meta.path.is_ident("display") {
                    let value = meta.value()?; 
                    let s: syn::LitStr = value.parse()?; 
                    
                    custom_repr_fn_name = Some(syn::Ident::new(&s.value(), s.span()));
                    return Ok(());
                }                
                Ok(())
            });
        }
    }
    
    // module display
    let extra_repr_fn = if let Some(fn_name) = custom_repr_fn_name {
        quote! {
            fn extra_repr(&self) -> String {
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
            use crate::modules::Module;
            write!(f, "{}", Module::<f64>::display(&*self))
        },
    };

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

    // codegen
    let codegen = quote! {
        impl #impl_generics_tokens #lumen::modules::Module<#module_generic_type> for #name #ty_generics #where_clause {
            
            fn visit_param<Visitor: #lumen::modules::ParamVisitor<#module_generic_type>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                #body
                Ok(())
            }

            fn visit_param_mut<Visitor: #lumen::modules::ParamVisitorMut<#module_generic_type>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                #body_mut
                Ok(())
            }

            fn visit_module<Visitor: #lumen::modules::ModuleVisitor<#module_generic_type>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                visitor.visit_module(self)?;
                #submodule_body
                visitor.visit_module_end(self)?;
                Ok(())
            }

            #extra_repr_fn
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

fn validate_and_extract_generic(generics: &syn::Generics) -> syn::Result<Option<&syn::Ident>> {
    let params: Vec<_> = generics.params.iter().collect();

    // 1. check count
    if params.len() > 1 {
        return Err(syn::Error::new_spanned(
            generics, 
            "Module struct allows at most one generic parameter (e.g., <T: FloatDType>)."
        ));
    }

    // no params
    if params.is_empty() {
        return Ok(None);
    }

    // 2. a param must be bounds by `FloatDType`
    let param = params[0];
    let type_param = match param {
        syn::GenericParam::Type(tp) => tp,
        _ => return Err(syn::Error::new_spanned(
            param, 
            "Module struct generic parameter must be a type parameter, not lifetime or const."
        )),
    };

    let has_float_dtype = type_param.bounds.iter().any(|bound| {
        if let syn::TypeParamBound::Trait(trait_bound) = bound {
            if let Some(segment) = trait_bound.path.segments.last() {
                return segment.ident == "FloatDType";
            }
        }
        false
    });

    if !has_float_dtype {
        return Err(syn::Error::new_spanned(
            type_param, 
            format!("The generic parameter '{}' must be bound by FloatDType (e.g., {}: FloatDType).", type_param.ident, type_param.ident)
        ));
    }

    Ok(Some(&type_param.ident))
}