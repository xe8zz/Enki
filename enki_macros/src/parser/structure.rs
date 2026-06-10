use syn::{DeriveInput, Data, Fields, Error, Attribute, LitInt};
use crate::parser::ir::{ParsedStruct, ParsedField, FieldType, SystemValue};

pub fn parse_enki_struct(input: &DeriveInput) -> Result<ParsedStruct, syn::Error> {
    let data_struct = match &input.data {
        Data::Struct(data) => data,
        _ => return Err(Error::new_spanned(
            input,
            "EnkiStruct can only be derived on standard structs with named fields."
        )),
    };

    let fields_named = match &data_struct.fields {
        Fields::Named(fields) => fields,
        _ => return Err(Error::new_spanned(
            &data_struct.fields,
            "EnkiStruct only supports standard structs with named fields (e.g., struct MyStruct { field: Type })."
        )),
    };

    let mut parsed_fields = Vec::with_capacity(fields_named.named.len());
    for field in &fields_named.named {
        let field_name = field.ident.as_ref()
            .ok_or_else(|| Error::new_spanned(field, "Field must have a valid identifier name"))?
            .to_string();

        let field_type = parse_type(&field.ty)?;

        let system_value = parse_field_attributes(&field.attrs)?;

        parsed_fields.push(ParsedField {
            name: field_name,
            ty: field_type,
            system_value,
        });
    }

    Ok(ParsedStruct {
        name: input.ident.to_string(),
        fields: parsed_fields,
    })
}
fn parse_field_attributes(attrs: &[Attribute]) -> Result<SystemValue, syn::Error> {
    let mut system_value = SystemValue::None;

    for attr in attrs {
        if attr.path().is_ident("enki") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("position") {
                    system_value = SystemValue::Position;
                } else if meta.path.is_ident("target") {
                    let value = meta.value()?;
                    let lit_int: LitInt = value.parse()?;
                    let target_val = lit_int.base10_parse::<u32>()?;
                    system_value = SystemValue::Target(target_val);
                } else if meta.path.is_ident("depth") {
                    system_value = SystemValue::Depth;
                } else if meta.path.is_ident("vertex_id") {
                    system_value = SystemValue::VertexId;
                } else if meta.path.is_ident("instance_id") {
                    system_value = SystemValue::InstanceId;
                } else if meta.path.is_ident("flat") {
                    system_value = SystemValue::Flat;
                } else if meta.path.is_ident("point_size") {
                    system_value = SystemValue::PointSize;
                } else {
                    return Err(meta.error("Unsupported parameter inside #[enki(...)] attribute"));
                }
                Ok(())
            })?;
        }
    }

    Ok(system_value)
}
fn parse_type(ty: &syn::Type) -> Result<FieldType, syn::Error> {
    match ty {
        syn::Type::Path(type_path) => {
            let segment = type_path.path.segments.first()
                .ok_or_else(|| Error::new_spanned(ty, "Invalid or empty type path"))?;
            let type_name = segment.ident.to_string();

            match type_name.as_str() {
                "f32" => Ok(FieldType::Float),
                "i32" => Ok(FieldType::Int),
                "u32" => Ok(FieldType::Uint),
                _ => Err(Error::new_spanned(
                    ty,
                    format!(
                        "Unsupported primitive GPU type: '{}'. Only f32, i32, and u32 are supported in EnkiStruct.",
                        type_name
                    )
                )),
            }
        }
        syn::Type::Array(type_array) => {
            let elem_type = parse_type(&type_array.elem)?;

            let len_expr = &type_array.len;
            let len_val = match len_expr {
                syn::Expr::Lit(expr_lit) => {
                    match &expr_lit.lit {
                        syn::Lit::Int(lit_int) => {
                            lit_int.base10_parse::<usize>()?
                        }
                        _ => return Err(Error::new_spanned(len_expr, "Array dimensions length must be an integer literal.")),
                    }
                }
                _ => return Err(Error::new_spanned(len_expr, "Array dimensions length must be a constant literal.")),
            };

            match (elem_type, len_val) {
                (FieldType::Float, 2) => Ok(FieldType::Float2),
                (FieldType::Float, 3) => Ok(FieldType::Float3),
                (FieldType::Float, 4) => Ok(FieldType::Float4),
                (FieldType::Int, 2) => Ok(FieldType::Int2),
                (FieldType::Int, 3) => Ok(FieldType::Int3),
                (FieldType::Int, 4) => Ok(FieldType::Int4),
                _ => Err(Error::new_spanned(
                    ty,
                    "Unsupported GPU vector dimensions. Supported array vector dimensions are 2, 3, or 4 (e.g., [f32; 3] or [i32; 4])."
                )),
            }
        }
        _ => Err(Error::new_spanned(
            ty,
            "Unsupported GPU memory type. GPU structures can only contain primitives (f32, i32, u32) or standard float/int vectors."
        )),
    }
}