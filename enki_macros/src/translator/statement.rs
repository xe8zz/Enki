use syn::{Stmt, Error, Pat};
use crate::translator::expression::translate_expr;
use crate::parser::ir::ShaderType;

pub fn translate_stmt(
    stmt: &Stmt,
    array_params: &[String],
    scalar_params: &[String],
    shader_type: ShaderType,
) -> Result<String, Error> {
    match stmt {
        Stmt::Local(local) => {
            let (var_name, is_mutable) = match &local.pat {
                Pat::Ident(pat_ident) => {
                    let name = pat_ident.ident.to_string();
                    let mutable = pat_ident.mutability.is_some();
                    (name, mutable)
                }
                _ => return Err(Error::new_spanned(
                    &local.pat,
                    "Unsupported variable pattern. Only simple variable names (e.g., let x = ... or let mut x = ...) are allowed on the GPU."
                )),
            };

            let init = local.init.as_ref()
                .ok_or_else(|| Error::new_spanned(
                    local,
                    "Variable declarations on the GPU must have an initial value (e.g., let x = 0.0;)."
                ))?;

            let expr_str = translate_expr(&init.expr, array_params, scalar_params, shader_type)?;

            if is_mutable {
                Ok(format!("var {} = {};", var_name, expr_str))
            } else {
                Ok(format!("let {} = {};", var_name, expr_str))
            }
        }

        Stmt::Expr(expr, semi) => {
            let expr_str = translate_expr(expr, array_params, scalar_params, shader_type)?;

            if semi.is_none() && shader_type != ShaderType::Compute {
                if expr_str.contains("return") {
                    Ok(expr_str)
                } else {
                    Ok(format!("return {};", expr_str))
                }
            } else if expr_str.ends_with(';') {
                Ok(expr_str)
            } else {
                Ok(format!("{};", expr_str))
            }
        }

        _ => Err(Error::new_spanned(
            stmt,
            "Unsupported statement inside GPU mathematical blocks. Only local variables (let/let mut) and assignments are allowed."
        )),
    }
}