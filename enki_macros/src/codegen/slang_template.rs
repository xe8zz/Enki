use crate::parser::ir::{ParsedFunction, ParsedParam, ShaderType, SystemValue};
use crate::registry::load_struct_from_cache;

pub fn generate_slang_source(func: &ParsedFunction, translated_body: &str) -> Result<String, String> {
    match func.shader_type {
        ShaderType::Compute => generate_compute_slang(func, translated_body),
        ShaderType::Vertex => generate_vertex_slang(func, translated_body),
        ShaderType::Fragment => generate_fragment_slang(func, translated_body),
    }
}

fn generate_compute_slang(func: &ParsedFunction, translated_body: &str) -> Result<String, String> {
    let mut slang_source = String::new();
    let mut declared_structs = std::collections::HashSet::new();

    for param in &func.params {
        if let ParsedParam::Array(_, struct_name) = param {
            if !declared_structs.contains(struct_name) {
                let parsed_struct = load_struct_from_cache(struct_name)?;
                slang_source.push_str(&format!("struct {} {{\n", parsed_struct.name));
                for field in parsed_struct.fields {
                    slang_source.push_str(&format!("    {} {};\n", field.ty.to_slang_type(), field.name));
                }
                slang_source.push_str("};\n\n");
                declared_structs.insert(struct_name.clone());
            }
        }
    }

    let mut register_idx = 0;
    for param in &func.params {
        if let ParsedParam::Array(param_name, struct_name) = param {
            slang_source.push_str(&format!(
                "RWStructuredBuffer<{}> {}[] : register(u0);\n",
                struct_name, param_name
            ));
            register_idx += 1;
        }
    }
    if register_idx > 0 {
        slang_source.push_str("\n");
    }

    let mut has_scalars = false;
    let mut pc_fields = String::new();
    for param in &func.params {
        if let ParsedParam::Scalar(param_name, ty) = param {
            pc_fields.push_str(&format!("    {} {};\n", ty.to_slang_type(), param_name));
            has_scalars = true;
        }
    }

    if has_scalars {
        slang_source.push_str("struct PushConstants {\n");
        slang_source.push_str(&pc_fields);
        slang_source.push_str("};\n");
        slang_source.push_str("ParameterBlock<PushConstants> pc;\n\n");
    }

    slang_source.push_str("[shader(\"compute\")]\n");
    slang_source.push_str("[numthreads(64, 1, 1)]\n");
    slang_source.push_str("void main(uint3 thread_idx : SV_DispatchThreadID) {\n");
    slang_source.push_str("    uint index = thread_idx.x;\n\n");

    slang_source.push_str(translated_body);
    slang_source.push_str("\n}\n");

    Ok(slang_source)
}

fn generate_vertex_slang(func: &ParsedFunction, translated_body: &str) -> Result<String, String> {
    let mut slang_source = String::new();

    if func.params.is_empty() {
        return Err("[Vertex Generator] Vertex Shader must have at least one input parameter representing vertex attributes.".to_string());
    }

    let first_param = &func.params[0];
    if let ParsedParam::Array(param_name, struct_name) = first_param {
        let parsed_struct = load_struct_from_cache(struct_name)?;
        slang_source.push_str(&format!("struct {} {{\n", parsed_struct.name));
        for (i, field) in parsed_struct.fields.iter().enumerate() {
            slang_source.push_str(&format!(
                "    [[vk::location({})]] {} {};\n",
                i,
                field.ty.to_slang_type(),
                field.name
            ));
        }
        slang_source.push_str("};\n\n");
    } else {
        return Err("[Vertex Generator] First parameter must be a custom struct representing vertex attributes.".to_string());
    }

    if let Some(ref ret_struct_name) = func.return_type {
        let parsed_struct = load_struct_from_cache(ret_struct_name)?;
        slang_source.push_str(&format!("struct {} {{\n", parsed_struct.name));
        for field in &parsed_struct.fields {
            let semantic = match field.system_value {
                SystemValue::Position => " : SV_Position".to_string(),
                SystemValue::Target(idx) => format!(" : SV_Target{}", idx),
                SystemValue::Depth => " : SV_Depth".to_string(),
                SystemValue::PointSize => " : SV_PointSize".to_string(),
                _ => "".to_string(),
            };
            let modifier = if field.system_value == SystemValue::Flat {
                "no_perspective "
            } else {
                ""
            };

            let slang_type = if field.system_value == SystemValue::Position {
                "float4".to_string()
            } else {
                field.ty.to_slang_type()
            };

            slang_source.push_str(&format!(
                "    {}{} {}{};\n",
                modifier,
                slang_type,
                field.name,
                semantic
            ));
        }
        slang_source.push_str("};\n\n");
    } else {
        return Err("[Vertex Generator] Vertex Shader must return a custom EnkiStruct containing position output.".to_string());
    }

    let mut ubo_idx = 0;
    for param in func.params.iter().skip(1) {
        if let ParsedParam::Array(param_name, struct_name) = param {
            let parsed_struct = load_struct_from_cache(struct_name)?;
            slang_source.push_str(&format!("struct {} {{\n", parsed_struct.name));
            for field in &parsed_struct.fields {
                slang_source.push_str(&format!("    {} {};\n", field.ty.to_slang_type(), field.name));
            }
            slang_source.push_str("};\n\n");

            slang_source.push_str(&format!(
                "ConstantBuffer<{}> {} : register(b{});\n\n",
                struct_name, param_name, ubo_idx
            ));
            ubo_idx += 1;
        }
    }

    slang_source.push_str("[shader(\"vertex\")]\n");
    if let Some(ref ret_struct_name) = func.return_type {
        if let ParsedParam::Array(param_name, struct_name) = first_param {
            slang_source.push_str(&format!(
                "{} main({} {}, uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID) {{\n",
                ret_struct_name, struct_name, param_name
            ));
        }
    }

    slang_source.push_str(translated_body);
    slang_source.push_str("\n}\n");

    Ok(slang_source)
}

fn generate_fragment_slang(func: &ParsedFunction, translated_body: &str) -> Result<String, String> {
    let mut slang_source = String::new();

    if func.params.is_empty() {
        return Err("[Fragment Generator] Fragment Shader must have at least one input parameter representing varying attributes from vertex shader.".to_string());
    }

    let first_param = &func.params[0];
    if let ParsedParam::Array(param_name, struct_name) = first_param {
        let parsed_struct = load_struct_from_cache(struct_name)?;
        slang_source.push_str(&format!("struct {} {{\n", parsed_struct.name));
        for field in &parsed_struct.fields {
            let semantic = match field.system_value {
                SystemValue::Position => " : SV_Position".to_string(),
                _ => "".to_string(),
            };
            let modifier = if field.system_value == SystemValue::Flat {
                "no_perspective "
            } else {
                ""
            };

            let slang_type = if field.system_value == SystemValue::Position {
                "float4".to_string()
            } else {
                field.ty.to_slang_type()
            };

            slang_source.push_str(&format!(
                "    {}{} {}{};\n",
                modifier,
                slang_type,
                field.name,
                semantic
            ));
        }
        slang_source.push_str("};\n\n");
    } else {
        return Err("[Fragment Generator] First parameter must be a custom struct representing varying vertex outputs.".to_string());
    }

    let mut ubo_idx = 0;
    for param in func.params.iter().skip(1) {
        if let ParsedParam::Array(param_name, struct_name) = param {
            let parsed_struct = load_struct_from_cache(struct_name)?;
            slang_source.push_str(&format!("struct {} {{\n", parsed_struct.name));
            for field in &parsed_struct.fields {
                slang_source.push_str(&format!("    {} {};\n", field.ty.to_slang_type(), field.name));
            }
            slang_source.push_str("};\n\n");

            slang_source.push_str(&format!(
                "ConstantBuffer<{}> {} : register(b{});\n\n",
                struct_name, param_name, ubo_idx
            ));
            ubo_idx += 1;
        }
    }

    let mut is_primitive_output = false;
    let mut slang_output_type = String::new();

    if let Some(ref ret_type) = func.return_type {
        if ret_type == "float4" || ret_type == "float3" || ret_type == "float2" {
            is_primitive_output = true;
            slang_output_type = ret_type.clone();
        } else {
            let parsed_struct = load_struct_from_cache(ret_type)?;
            slang_source.push_str(&format!("struct {} {{\n", parsed_struct.name));
            for field in &parsed_struct.fields {
                let semantic = match field.system_value {
                    SystemValue::Target(idx) => format!(" : SV_Target{}", idx),
                    SystemValue::Depth => " : SV_Depth".to_string(),
                    _ => " : SV_Target0".to_string(),
                };
                slang_source.push_str(&format!(
                    "    {} {}{};\n",
                    field.ty.to_slang_type(),
                    field.name,
                    semantic
                ));
            }
            slang_source.push_str("};\n\n");
            slang_output_type = ret_type.clone();
        }
    } else {
        is_primitive_output = true;
        slang_output_type = "float4".to_string();
    }

    slang_source.push_str("[shader(\"fragment\")]\n");
    if let ParsedParam::Array(param_name, struct_name) = first_param {
        if is_primitive_output {
            slang_source.push_str(&format!(
                "{} main({} {}) : SV_Target0 {{\n",
                slang_output_type, struct_name, param_name
            ));
        } else {
            slang_source.push_str(&format!(
                "{} main({} {}) {{\n",
                slang_output_type, struct_name, param_name
            ));
        }
    }

    slang_source.push_str(translated_body);
    slang_source.push_str("\n}\n");

    Ok(slang_source)
}