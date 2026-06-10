use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemValue {
    Position,
    Target(u32),
    Depth,
    VertexId,
    InstanceId,
    Flat,
    PointSize,
    None,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    Float,
    Int,
    Uint,
    Float2,
    Float3,
    Float4,
    Int2,
    Int3,
    Int4,
    Custom(String),
}

impl FieldType {
    pub fn to_slang_type(&self) -> String {
        match self {
            FieldType::Float => "float".to_string(),
            FieldType::Int => "int".to_string(),
            FieldType::Uint => "uint".to_string(),
            FieldType::Float2 => "float2".to_string(),
            FieldType::Float3 => "float3".to_string(),
            FieldType::Float4 => "float4".to_string(),
            FieldType::Int2 => "int2".to_string(),
            FieldType::Int3 => "int3".to_string(),
            FieldType::Int4 => "int4".to_string(),
            FieldType::Custom(name) => name.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedField {
    pub name: String,
    pub ty: FieldType,
    pub system_value: SystemValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedStruct {
    pub name: String,
    pub fields: Vec<ParsedField>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderType {
    Compute,
    Fragment,
    Vertex,
}

#[derive(Debug, Clone)]
pub enum ParsedParam {
    Array(String, String),
    Scalar(String, FieldType),
}

#[derive(Debug, Clone)]
pub struct ParsedFunction {
    pub name: String,
    pub shader_type: ShaderType,
    pub params: Vec<ParsedParam>,
    pub block: syn::Block,
    pub return_type: Option<String>,
}