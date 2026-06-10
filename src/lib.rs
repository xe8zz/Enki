use std::sync::OnceLock;

pub use utu::compute::{ComputeEngine, ComputeEngineConfig};
pub use utu::graphics::{GpuWindow, GpuRenderer, GpuGraphicsPipeline};
pub use apsu::{GpuDeviceBuffer, GpuTransferManager, BufferUsage};
pub use enki_macros::{enki_compute, enki_vertex, enki_fragment, EnkiStruct};

static ENGINE: OnceLock<ComputeEngine> = OnceLock::new();

pub fn init_global_engine(config: ComputeEngineConfig) -> Result<&'static ComputeEngine, String> {
    if let Some(engine) = ENGINE.get() {
        return Ok(engine);
    }

    let engine = ComputeEngine::new(config)
        .map_err(|e| format!("[Enki] Failed to initialize global compute engine: {}", e))?;

    match ENGINE.set(engine) {
        Ok(()) => Ok(ENGINE.get().unwrap()),
        Err(_) => Ok(ENGINE.get().unwrap()),
    }
}

pub fn get_global_engine() -> Result<&'static ComputeEngine, String> {
    ENGINE.get()
        .ok_or_else(|| "[Enki] Global compute engine is not initialized. Please call enki::init_global_engine() first.".to_string())
}