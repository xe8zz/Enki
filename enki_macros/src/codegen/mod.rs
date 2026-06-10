pub mod slang_template;
pub mod compiler;
pub mod rust_emitter;

use proc_macro2::TokenStream;
use syn::Error;
use crate::parser::ir::ParsedFunction;

pub fn compile_and_emit_compute(func: &ParsedFunction, translated_body: &str) -> Result<TokenStream, syn::Error> {
    let slang_source = slang_template::generate_slang_source(func, translated_body)
        .map_err(|e| Error::new(proc_macro2::Span::call_site(), e))?;

    eprintln!("================== GENERATED SLANG SHADER ==================\n{}\n============================================================", slang_source);

    let spv_bytes = compiler::compile_slang_to_spirv(&slang_source, &func.name)
        .map_err(|e| Error::new(proc_macro2::Span::call_site(), e))?;

    let rust_tokens = rust_emitter::emit_rust_wrapper(func, &spv_bytes);

    Ok(rust_tokens)
}