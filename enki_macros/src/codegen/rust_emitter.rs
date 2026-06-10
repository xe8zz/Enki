use proc_macro2::{TokenStream, Ident, Span};
use quote::quote;
use crate::parser::ir::{ParsedFunction, ParsedParam, FieldType};

pub fn emit_rust_wrapper(func: &ParsedFunction, spv_bytes: &[u8]) -> TokenStream {
    let func_name = Ident::new(&func.name, Span::call_site());

    let spv_literal = proc_macro2::Literal::byte_string(spv_bytes);

    let mut array_names = Vec::new();
    let mut array_structs = Vec::new();
    let mut array_indices = Vec::new();

    let mut scalar_names = Vec::new();
    let mut scalar_types = Vec::new();

    for param in &func.params {
        match param {
            ParsedParam::Array(name, struct_name) => {
                array_indices.push(array_names.len() as u32);
                array_names.push(Ident::new(name, Span::call_site()));
                array_structs.push(Ident::new(struct_name, Span::call_site()));
            }
            ParsedParam::Scalar(name, ty) => {
                scalar_names.push(Ident::new(name, Span::call_site()));
                scalar_types.push(map_field_type_to_rust_token(ty));
            }
        }
    }

    let first_array_name = &array_names[0];

    let pcs_gen = if scalar_names.is_empty() {
        quote! {
            let pcs_bytes: &[u8] = &[];
        }
    } else {
        quote! {
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            #[repr(C)]
            struct PushConstants {
                #(#scalar_names: #scalar_types,)*
            }

            let pcs = PushConstants {
                #(#scalar_names,)*
            };
            let pcs_bytes = bytemuck::bytes_of(&pcs);
        }
    };

    quote! {
        pub fn #func_name(
            #(#array_names: &apsu::GpuDeviceBuffer<#array_structs>,)*
            #(#scalar_names: #scalar_types,)*
        ) -> Result<(), String> {
            use std::sync::OnceLock;

            const SPV_BYTES: &[u8] = #spv_literal;

            static PIPELINE: OnceLock<utu::compute::ComputePipeline> = OnceLock::new();

            let engine = enki::get_global_engine()
                .map_err(|e| format!("[Enki Engine Error] {}", e))?;

            let pipeline = PIPELINE.get_or_init(|| {
                utu::compute::ComputePipeline::new(&engine, SPV_BYTES)
                    .expect("[Enki Codegen] Failed to compile dynamic GPU compute pipeline")
            });

            #(
                engine.bind_storage_device_buffer(#array_indices, #array_names)
                    .map_err(|e| format!("[Enki Bind Error] {}", e))?;
            )*

            #pcs_gen

            let executor = utu::compute::ComputeExecutor::new(&engine);
            let element_count = #first_array_name.element_count;
            let grid_x = (element_count as u32 + 63) / 64;

            executor.dispatch(pipeline, (grid_x, 1, 1), pcs_bytes, true)
                .map_err(|e| format!("[Enki Dispatch Error] {}", e))?;

            Ok(())
        }
    }
}

fn map_field_type_to_rust_token(ty: &FieldType) -> TokenStream {
    match ty {
        FieldType::Float => quote! { f32 },
        FieldType::Int => quote! { i32 },
        FieldType::Uint => quote! { u32 },
        FieldType::Float2 => quote! { [f32; 2] },
        FieldType::Float3 => quote! { [f32; 3] },
        FieldType::Float4 => quote! { [f32; 4] },
        FieldType::Int2 => quote! { [i32; 2] },
        FieldType::Int3 => quote! { [i32; 3] },
        FieldType::Int4 => quote! { [i32; 4] },
        FieldType::Custom(name) => {
            let ident = Ident::new(name, Span::call_site());
            quote! { #ident }
        }
    }
}