use proc_macro2::TokenStream;
use quote::quote;
use crate::utils;

use super::common;

pub fn generate_enum(ast: &syn::DeriveInput) -> TokenStream {
    let lumen = utils::get_lumen_nn_path();
    let name = &ast.ident;

    // extract generic param (复用你的解析逻辑)
    let validated_generic = match common::validate_and_extract_generic(&ast.generics) {
        Ok(res) => res,
        Err(e) => return e.to_compile_error().into(),
    };

    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let (impl_generics_tokens, module_generic_type) = match validated_generic.as_ref() {
        Some(user_generic_ident) => (quote! { #impl_generics }, quote! { #user_generic_ident }),
        None => (quote! { <T: lumen_core::FloatDType> }, quote! { T }),
    };

    // module attr (复用你的属性解析逻辑)
    let mut custom_display_fn_name: Option<syn::Ident> = None;
    let mut custom_set_train_name: Option<syn::Ident> = None;

    for attr in &ast.attrs {
        if attr.path().is_ident("module") {
            let _ = attr.parse_nested_meta(|meta| {
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
        quote! { fn extra_display(&self) -> String { self.#fn_name() } }
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

    // set train
    let set_train_fn = if let Some(fn_name) = custom_set_train_name {
        quote! { fn set_train(&mut self, mode: bool) { self.#fn_name(mode) } }
    } else {
        quote! {}
    };

    // 为 Enum 生成 match 语句的分支
    let mut param_arms = quote! {};
    let mut param_mut_arms = quote! {};
    let mut state_arms = quote! {};
    let mut state_mut_arms = quote! {};
    let mut buffer_arms = quote! {};
    let mut buffer_mut_arms = quote! {};
    let mut module_arms = quote! {};
    let mut module_mut_arms = quote! {};

    if let syn::Data::Enum(enum_data) = &ast.data {
        for variant in enum_data.variants.iter() {
            let variant_ident = &variant.ident;
            let variant_name_str = variant_ident.to_string();

            // 检查是否有 #[module(skip)]
            let should_skip = variant.attrs.iter().any(|attr| {
                if !attr.path().is_ident("module") { return false; }
                let mut found_skip = false;
                let _ = attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("skip") { found_skip = true; }
                    Ok(())
                });
                found_skip
            });

            // 如果该变体被跳过，则生成空的匹配分支以保证 match 穷尽性
            if should_skip {
                let empty_arm = quote! { Self::#variant_ident(_) => {} };
                // let empty_arm_res = quote! { Self::#variant_ident(_) => Ok(()) };
                
                param_arms.extend(empty_arm.clone());
                param_mut_arms.extend(empty_arm.clone());
                buffer_arms.extend(empty_arm.clone());
                buffer_mut_arms.extend(empty_arm.clone());
                state_arms.extend(empty_arm.clone());
                state_mut_arms.extend(empty_arm.clone());
                module_arms.extend(empty_arm.clone());
                module_mut_arms.extend(empty_arm.clone());
                continue;
            }

            // 确保这是一个单字段的元组变体 (例如 Gelu(Gelu))
            match &variant.fields {
                syn::Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {}
                _ => return syn::Error::new_spanned(
                    variant,
                    "Enum variants in a Module must have exactly one unnamed field (e.g., `Gelu(Gelu)`)."
                ).to_compile_error().into(),
            }

            // 构造分支代码 (Rust match ergonomics 会自动处理 & 和 &mut)
            param_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner);
                    #lumen::modules::Module::visit_param(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner);
                }
            });

            param_mut_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner);
                    #lumen::modules::Module::visit_param_mut(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner);
                }
            });

            buffer_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner);
                    #lumen::modules::Module::visit_buffer(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner);
                }
            });

            buffer_mut_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner);
                    #lumen::modules::Module::visit_buffer_mut(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner);
                }
            });

            state_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner);
                    #lumen::modules::Module::visit_state(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner);
                }
            });

            state_mut_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner);
                    #lumen::modules::Module::visit_state_mut(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner);
                }
            });

            module_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner)?;
                    #lumen::modules::Module::visit_module(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner)?;
                }
            });

            module_mut_arms.extend(quote! {
                Self::#variant_ident(inner) => {
                    visitor.enter_submodule(#variant_name_str, inner)?;
                    #lumen::modules::Module::visit_module_mut(inner, visitor)?;
                    visitor.exit_submodule(#variant_name_str, inner)?;
                }
            });
        }
    }

    // codegen
    let codegen = quote! {
        impl #impl_generics_tokens #lumen::modules::Module<#module_generic_type> for #name #ty_generics #where_clause {
            
            fn visit_param<Visitor: #lumen::modules::TensorVisitor<#module_generic_type>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                match self { #param_arms }
                Ok(())
            }

            fn visit_param_mut<Visitor: #lumen::modules::TensorVisitorMut<#module_generic_type>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                match self { #param_mut_arms }
                Ok(())
            }

            fn visit_buffer<Visitor: #lumen::modules::TensorVisitor<#module_generic_type>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                match self { #buffer_arms }
                Ok(())
            }

            fn visit_buffer_mut<Visitor: #lumen::modules::TensorVisitorMut<#module_generic_type>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                match self { #buffer_mut_arms }
                Ok(())
            }

            fn visit_state<Visitor: #lumen::modules::TensorVisitor<#module_generic_type>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                match self { #state_arms }
                Ok(())
            }

            fn visit_state_mut<Visitor: #lumen::modules::TensorVisitorMut<#module_generic_type>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                match self { #state_mut_arms }
                Ok(())
            }

            fn visit_module<Visitor: #lumen::modules::ModuleVisitor<#module_generic_type>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                visitor.visit_module(self)?;
                match self { #module_arms }
                visitor.visit_module_end(self)?;
                Ok(())
            }

            fn visit_module_mut<Visitor: #lumen::modules::ModuleVisitorMut<#module_generic_type>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
                visitor.visit_module_mut(self)?;
                match self { #module_mut_arms }
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

    codegen
}