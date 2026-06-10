use syn::{Expr, Error, BinOp};
use crate::parser::ir::ShaderType;
use crate::registry::load_struct_from_cache;

pub fn translate_expr(
    expr: &Expr,
    array_params: &[String],
    scalar_params: &[String],
    shader_type: ShaderType,
) -> Result<String, Error> {
    match expr {
        Expr::Path(expr_path) => {
            let segment = expr_path.path.segments.last()
                .ok_or_else(|| Error::new_spanned(expr, "Invalid path"))?;
            let ident_str = segment.ident.to_string();

            if scalar_params.contains(&ident_str) {
                Ok(format!("pc.{}", ident_str))
            } else if array_params.contains(&ident_str) {
                if shader_type == ShaderType::Compute {
                    let slot_idx = array_params.iter().position(|p| p == &ident_str).unwrap_or(0);
                    Ok(format!("{}[{}][index]", ident_str, slot_idx))
                } else {
                    Ok(ident_str)
                }
            } else {
                Ok(ident_str)
            }
        }

        Expr::Lit(expr_lit) => {
            match &expr_lit.lit {
                syn::Lit::Float(lit_float) => Ok(lit_float.to_string()),
                syn::Lit::Int(lit_int) => Ok(lit_int.to_string()),
                syn::Lit::Bool(lit_bool) => Ok(lit_bool.value.to_string()),
                _ => Err(Error::new_spanned(expr, "Unsupported literal type on the GPU")),
            }
        }

        Expr::Binary(expr_binary) => {
            let left = translate_expr(&expr_binary.left, array_params, scalar_params, shader_type)?;
            let right = translate_expr(&expr_binary.right, array_params, scalar_params, shader_type)?;
            let op = translate_bin_op(&expr_binary.op);
            if op.is_empty() {
                return Err(Error::new_spanned(&expr_binary.op, "Unsupported binary operator on the GPU"));
            }
            Ok(format!("({} {} {})", left, op, right))
        }

        Expr::Field(expr_field) => {
            let base = translate_expr(&expr_field.base, array_params, scalar_params, shader_type)?;
            let field = match &expr_field.member {
                syn::Member::Named(ident) => ident.to_string(),
                _ => return Err(Error::new_spanned(expr, "Tuple indices are not supported on the GPU")),
            };
            Ok(format!("{}.{}", base, field))
        }

        Expr::MethodCall(expr_method) => {
            let base = translate_expr(&expr_method.receiver, array_params, scalar_params, shader_type)?;
            let method_name = expr_method.method.to_string();

            let mut args = Vec::with_capacity(expr_method.args.len());
            for arg in &expr_method.args {
                args.push(translate_expr(arg, array_params, scalar_params, shader_type)?);
            }

            if let Some(mapped) = crate::translator::intrinsic::map_rust_method_to_slang(&method_name, &base, &args) {
                Ok(mapped)
            } else {
                Err(Error::new_spanned(&expr_method.method, format!("Unsupported GPU math method: '{}'", method_name)))
            }
        }

        Expr::Call(expr_call) => {
            let func_name = match &*expr_call.func {
                Expr::Path(expr_path) => {
                    let segment = expr_path.path.segments.last()
                        .ok_or_else(|| Error::new_spanned(&expr_call.func, "Invalid function call path"))?;
                    segment.ident.to_string()
                }
                _ => return Err(Error::new_spanned(&expr_call.func, "Unsupported function call expression")),
            };

            let mut args = Vec::with_capacity(expr_call.args.len());
            for arg in &expr_call.args {
                args.push(translate_expr(arg, array_params, scalar_params, shader_type)?);
            }

            if let Some(mapped) = crate::translator::intrinsic::map_rust_function_to_slang(&func_name, &args) {
                Ok(mapped)
            } else {
                Err(Error::new_spanned(&expr_call.func, format!("Unsupported GPU math function: '{}'", func_name)))
            }
        }

        Expr::Assign(expr_assign) => {
            let left = translate_expr(&expr_assign.left, array_params, scalar_params, shader_type)?;
            let right = translate_expr(&expr_assign.right, array_params, scalar_params, shader_type)?;
            Ok(format!("{} = {};", left, right))
        }

        Expr::Paren(expr_paren) => {
            let inner = translate_expr(&expr_paren.expr, array_params, scalar_params, shader_type)?;
            Ok(format!("({})", inner))
        }

        Expr::Unary(expr_unary) => {
            let inner = translate_expr(&expr_unary.expr, array_params, scalar_params, shader_type)?;
            let op = match expr_unary.op {
                syn::UnOp::Neg(_) => "-",
                syn::UnOp::Not(_) => "!",
                _ => return Err(Error::new_spanned(&expr_unary.op, "Unsupported unary operator on the GPU")),
            };
            Ok(format!("{}{}", op, inner))
        }

        Expr::Index(expr_index) => {
            let base = translate_expr(&expr_index.expr, array_params, scalar_params, shader_type)?;
            let index = translate_expr(&expr_index.index, array_params, scalar_params, shader_type)?;
            Ok(format!("{}[{}]", base, index))
        }

        Expr::Array(expr_array) => {
            let mut elems = Vec::new();
            for elem in &expr_array.elems {
                elems.push(translate_expr(elem, array_params, scalar_params, shader_type)?);
            }
            let len = elems.len();
            Ok(format!("float{}({})", len, elems.join(", ")))
        }

        Expr::Struct(expr_struct) => {
            let struct_name = expr_struct.path.segments.last()
                .ok_or_else(|| Error::new_spanned(expr, "Invalid struct path"))?
                .ident.to_string();

            let parsed_struct = load_struct_from_cache(&struct_name)
                .map_err(|e| Error::new_spanned(expr, e))?;

            let mut slang_assignments = Vec::new();
            slang_assignments.push(format!("    {} _tmp;", struct_name));

            for field in &parsed_struct.fields {
                let user_field = expr_struct.fields.iter().find(|f| {
                    if let syn::Member::Named(ident) = &f.member {
                        ident.to_string() == field.name
                    } else {
                        false
                    }
                });

                if let Some(uf) = user_field {
                    let val_str = translate_expr(&uf.expr, array_params, scalar_params, shader_type)?;

                    let final_val = if field.system_value == crate::parser::ir::SystemValue::Position {
                        match field.ty {
                            crate::parser::ir::FieldType::Float2 => format!("float4({}, 0.0, 1.0)", val_str),
                            crate::parser::ir::FieldType::Float3 => format!("float4({}, 1.0)", val_str),
                            _ => val_str,
                        }
                    } else {
                        val_str
                    };

                    slang_assignments.push(format!("    _tmp.{} = {};", field.name, final_val));
                }
            }

            slang_assignments.push("    return _tmp;".to_string());
            Ok(slang_assignments.join("\n"))
        }

        _ => Err(Error::new_spanned(expr, "This statement or expression is too complex to compile to the GPU. Keep math blocks simple.")),
    }
}

fn translate_bin_op(op: &BinOp) -> &str {
    match op {
        BinOp::Add(_) => "+",
        BinOp::Sub(_) => "-",
        BinOp::Mul(_) => "*",
        BinOp::Div(_) => "/",
        BinOp::Rem(_) => "%",
        BinOp::And(_) => "&&",
        BinOp::Or(_) => "||",
        BinOp::Eq(_) => "==",
        BinOp::Lt(_) => "<",
        BinOp::Le(_) => "<=",
        BinOp::Gt(_) => ">",
        BinOp::Ge(_) => ">=",
        BinOp::Ne(_) => "!=",

        BinOp::AddAssign(_) => "+=",
        BinOp::SubAssign(_) => "-=",
        BinOp::MulAssign(_) => "*=",
        BinOp::DivAssign(_) => "/=",
        BinOp::RemAssign(_) => "%=",
        _ => "",
    }
}