use std::path::{Path, PathBuf};
use std::fs;
use crate::parser::ir::ParsedStruct;

pub fn get_cache_dir() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    let workspace_root = manifest_dir.parent()
        .unwrap_or(manifest_dir);

    workspace_root.join("target").join("enki_cache")
}

pub fn save_struct_to_cache(parsed: &ParsedStruct) -> Result<(), String> {
    let cache_dir = get_cache_dir();

    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir)
            .map_err(|e| format!("[Registry] Failed to create enki cache directory: {}", e))?;
    }

    let file_path = cache_dir.join(format!("{}.json", parsed.name));

    let json_str = serde_json::to_string_pretty(parsed)
        .map_err(|e| format!("[Registry] Failed to serialize struct metadata: {}", e))?;

    fs::write(&file_path, json_str)
        .map_err(|e| format!("[Registry] Failed to write struct metadata to file {:?}: {}", file_path, e))?;

    Ok(())
}

pub fn load_struct_from_cache(struct_name: &str) -> Result<ParsedStruct, String> {
    let cache_dir = get_cache_dir();
    let file_path = cache_dir.join(format!("{}.json", struct_name));

    if !file_path.exists() {
        return Err(format!(
            "[Registry] Struct '{}' metadata not found in compiler cache.\n\
             Help: Did you annotate the struct with #[derive(EnkiStruct)]?",
            struct_name
        ));
    }

    let json_str = fs::read_to_string(&file_path)
        .map_err(|e| format!("[Registry] Failed to read cached metadata file {:?}: {}", file_path, e))?;

    let parsed: ParsedStruct = serde_json::from_str(&json_str)
        .map_err(|e| format!("[Registry] Failed to deserialize cached struct metadata: {}", e))?;

    Ok(parsed)
}