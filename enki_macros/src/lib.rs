extern crate proc_macro;

use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput, ItemFn};
use quote::quote;

mod parser;
mod registry;
mod translator;
mod codegen;

use parser::ir::ShaderType;

#[proc_macro_derive(EnkiStruct, attributes(enki))]
pub fn enki_struct_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let parsed_struct = match parser::parse_enki_struct(&input) {
        Ok(s) => s,
        Err(e) => return e.to_compile_error().into(),
    };

    if let Err(e) = registry::save_struct_to_cache(&parsed_struct) {
        let syn_err = syn::Error::new(input.ident.span(), e);
        return syn_err.to_compile_error().into();
    }

    let expanded = quote! {};
    TokenStream::from(expanded)
}

#[proc_macro_attribute]
pub fn enki_compute(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_fn = parse_macro_input!(item as ItemFn);

    let parsed_function = match parser::parse_enki_function(&item_fn, ShaderType::Compute) {
        Ok(f) => f,
        Err(e) => return e.to_compile_error().into(),
    };

    let translated_body = match translator::translate_function_body(&parsed_function) {
        Ok(body) => body,
        Err(e) => return e.to_compile_error().into(),
    };

    let expanded = match codegen::compile_and_emit_compute(&parsed_function, &translated_body) {
        Ok(tokens) => tokens,
        Err(e) => return e.to_compile_error().into(),
    };

    TokenStream::from(expanded)
}

#[proc_macro_attribute]
pub fn enki_vertex(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_fn = parse_macro_input!(item as ItemFn);

    let parsed_function = match parser::parse_enki_function(&item_fn, ShaderType::Vertex) {
        Ok(f) => f,
        Err(e) => return e.to_compile_error().into(),
    };

    let translated_body = match translator::translate_function_body(&parsed_function) {
        Ok(body) => body,
        Err(e) => return e.to_compile_error().into(),
    };

    let slang_source = match codegen::slang_template::generate_slang_source(&parsed_function, &translated_body) {
        Ok(s) => s,
        Err(e) => return syn::Error::new(proc_macro2::Span::call_site(), e).to_compile_error().into(),
    };

    let spv_bytes = match codegen::compiler::compile_slang_to_spirv(&slang_source, &parsed_function.name) {
        Ok(b) => b,
        Err(e) => return syn::Error::new(proc_macro2::Span::call_site(), e).to_compile_error().into(),
    };

    let func_ident = syn::Ident::new(&parsed_function.name, proc_macro2::Span::call_site());
    let spv_literal = proc_macro2::Literal::byte_string(&spv_bytes);

    let expanded = quote! {
        pub fn #func_ident() -> &'static [u8] {
            const SPV_BYTES: &[u8] = #spv_literal;
            SPV_BYTES
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_attribute]
pub fn enki_fragment(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_fn = parse_macro_input!(item as ItemFn);

    let parsed_function = match parser::parse_enki_function(&item_fn, ShaderType::Fragment) {
        Ok(f) => f,
        Err(e) => return e.to_compile_error().into(),
    };

    let translated_body = match translator::translate_function_body(&parsed_function) {
        Ok(body) => body,
        Err(e) => return e.to_compile_error().into(),
    };

    let slang_source = match codegen::slang_template::generate_slang_source(&parsed_function, &translated_body) {
        Ok(s) => s,
        Err(e) => return syn::Error::new(proc_macro2::Span::call_site(), e).to_compile_error().into(),
    };

    let spv_bytes = match codegen::compiler::compile_slang_to_spirv(&slang_source, &parsed_function.name) {
        Ok(b) => b,
        Err(e) => return syn::Error::new(proc_macro2::Span::call_site(), e).to_compile_error().into(),
    };

    let func_ident = syn::Ident::new(&parsed_function.name, proc_macro2::Span::call_site());
    let spv_literal = proc_macro2::Literal::byte_string(&spv_bytes);

    let expanded = quote! {
        pub fn #func_ident() -> &'static [u8] {
            const SPV_BYTES: &[u8] = #spv_literal;
            SPV_BYTES
        }
    };

    TokenStream::from(expanded)
}