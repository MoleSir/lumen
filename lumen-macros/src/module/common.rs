
pub fn validate_and_extract_generic(generics: &syn::Generics) -> syn::Result<Option<&syn::Ident>> {
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