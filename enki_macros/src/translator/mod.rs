pub mod intrinsic;
pub mod expression;
pub mod statement;

use crate::parser::ir::{ParsedFunction, ParsedParam};
use statement::translate_stmt;

pub fn translate_function_body(func: &ParsedFunction) -> Result<String, syn::Error> {
    let array_params: Vec<String> = func.params.iter()
        .filter_map(|p| match p {
            ParsedParam::Array(param_name, _) => Some(param_name.clone()),
            _ => None,
        })
        .collect();

    let scalar_params: Vec<String> = func.params.iter()
        .filter_map(|p| match p {
            ParsedParam::Scalar(param_name, _) => Some(param_name.clone()),
            _ => None,
        })
        .collect();

    let mut translated_lines = Vec::with_capacity(func.block.stmts.len());

    for stmt in &func.block.stmts {
        let translated_line = translate_stmt(stmt, &array_params, &scalar_params, func.shader_type)?;
        translated_lines.push(format!("    {}", translated_line));
    }

    let slang_body = translated_lines.join("\n");
    Ok(slang_body)
}