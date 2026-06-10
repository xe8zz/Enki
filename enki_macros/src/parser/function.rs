use syn::{ItemFn, FnArg, Pat, Type, Error, ReturnType};
use crate::parser::ir::{ParsedFunction, ParsedParam, ShaderType, FieldType};


pub fn parse_enki_function(
    item: &ItemFn,
    shader_type: ShaderType,
) -> Result<ParsedFunction, Error> {
    let name = item.sig.ident.to_string();
    let mut params = Vec::with_capacity(item.sig.inputs.len());

    for input in &item.sig.inputs {
        match input {
            FnArg::Receiver(_) => {
                return Err(Error::new_spanned(
                    input,
                    "Enki GPU kernels do not support 'self' parameters. They must be regular static or global functions."
                ));
            }
            FnArg::Typed(pat_type) => {
                let param_name = match &*pat_type.pat {
                    Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
                    _ => return Err(Error::new_spanned(
                        &pat_type.pat,
                        "Unsupported parameter pattern. Only simple, direct variable names are supported on the GPU."
                    )),
                };

                match &*pat_type.ty {
                    Type::Path(type_path) => {
                        let segment = type_path.path.segments.first()
                            .ok_or_else(|| Error::new_spanned(&pat_type.ty, "Invalid or empty parameter type path"))?;
                        let type_name = segment.ident.to_string();

                        match type_name.as_str() {
                            "f32" => params.push(ParsedParam::Scalar(param_name, FieldType::Float)),
                            "i32" => params.push(ParsedParam::Scalar(param_name, FieldType::Int)),
                            "u32" => params.push(ParsedParam::Scalar(param_name, FieldType::Uint)),

                            struct_name => {
                                params.push(ParsedParam::Array(param_name, struct_name.to_string()));
                            }
                        }
                    }
                    _ => return Err(Error::new_spanned(
                        &pat_type.ty,
                        "Unsupported GPU function parameter type. Must be a primitive (f32, i32, u32) or a custom EnkiStruct."
                    )),
                }
            }
        }
    }
    let return_type = match &item.sig.output {
        ReturnType::Default => None,
        ReturnType::Type(_, ty) => {
            match &**ty {
                Type::Path(type_path) => {
                    let segment = type_path.path.segments.first()
                        .ok_or_else(|| Error::new_spanned(ty, "Invalid return type path"))?;
                    Some(segment.ident.to_string())
                }
                Type::Array(type_array) => {
                    if let Type::Path(type_path) = &*type_array.elem {
                        let segment = type_path.path.segments.first().unwrap();
                        let elem_name = segment.ident.to_string();
                        if elem_name == "f32" {
                            if let syn::Expr::Lit(expr_lit) = &type_array.len {
                                if let syn::Lit::Int(lit_int) = &expr_lit.lit {
                                    let len = lit_int.base10_parse::<usize>().unwrap_or(0);
                                    match len {
                                        2 => Some("float2".to_string()),
                                        3 => Some("float3".to_string()),
                                        4 => Some("float4".to_string()),
                                        _ => None,
                                    }
                                } else { None }
                            } else { None }
                        } else { None }
                    } else { None }
                }
                _ => None,
            }
        }
    };

    Ok(ParsedFunction {
        name,
        shader_type,
        params,
        block: (*item.block).clone(),
        return_type,
    })
}