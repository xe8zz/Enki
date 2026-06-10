use std::fs;
use std::process::Command;
use std::path::Path;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use crate::registry::get_cache_dir;

fn calculate_hash(source: &str) -> String {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

fn atomic_write(final_path: &Path, data: &[u8]) -> std::io::Result<()> {
    let parent = final_path.parent().unwrap_or(final_path);
    let temp_name = format!("{}.tmp", final_path.file_name().unwrap().to_string_lossy());
    let temp_path = parent.join(temp_name);
    fs::write(&temp_path, data)?;
    fs::rename(&temp_path, final_path)?;
    Ok(())
}

pub fn compile_slang_to_spirv(slang_source: &str, function_name: &str) -> Result<Vec<u8>, String> {
    let cache_dir = get_cache_dir();

    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir)
            .map_err(|e| format!("[Compiler] Failed to create cache directory: {}", e))?;
    }

    let slang_path = cache_dir.join(format!("{}.slang", function_name));
    let spv_path = cache_dir.join(format!("{}.spv", function_name));
    let hash_path = cache_dir.join(format!("{}.hash", function_name));

    let current_hash = calculate_hash(slang_source);

    if hash_path.exists() && spv_path.exists() {
        if let Ok(cached_hash) = fs::read_to_string(&hash_path) {
            if cached_hash.trim() == current_hash {
                if let Ok(spv_bytes) = fs::read(&spv_path) {
                    eprintln!("[Compiler] Cache HIT for shader '{}'. Skipping slangc execution.", function_name);
                    return Ok(spv_bytes);
                }
            }
        }
    }

    eprintln!("[Compiler] Cache MISS for shader '{}'. Spawning 'slangc' process...", function_name);

    atomic_write(&slang_path, slang_source.as_bytes())
        .map_err(|e| format!("[Compiler] Failed to write Slang file atomically: {}", e))?;

    let output_res = Command::new("slangc")
        .arg(&slang_path)
        .arg("-target")
        .arg("spirv")
        .arg("-entry")
        .arg("main")
        .arg("-o")
        .arg(&spv_path)
        .output();

    let output = match output_res {
        Ok(out) => out,
        Err(e) => {
            return Err(format!(
                "[Compiler] Failed to execute 'slangc' binary. Ensure Vulkan SDK is installed and 'slangc' is in your system PATH.\n\
                 System Error Details: {}", e
            ));
        }
    };

    if !output.status.success() {
        let stderr_msg = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "[Compiler] Slang compilation failed for function '{}'.\n\
             Slangc Output Compiler Error:\n{}\n\
             Help: You can inspect and debug the generated shader code at: {:?}",
            function_name, stderr_msg, slang_path
        ));
    }

    let spv_bytes = fs::read(&spv_path)
        .map_err(|e| format!("[Compiler] Failed to read compiled SPIR-V binary from disk: {}", e))?;

    atomic_write(&hash_path, current_hash.as_bytes())
        .map_err(|e| format!("[Compiler] Failed to write shader hash file atomically: {}", e))?;


    Ok(spv_bytes)
}