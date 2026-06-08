use ash::vk;
use anyhow::{Result, Context};
use std::path::Path;
use std::fs::File;
use std::io::Cursor;

pub struct VulkanShaderModule {
    pub handle: vk::ShaderModule,
    device: ash::Device,
}

impl VulkanShaderModule {
    pub fn from_words(device: &ash::Device, words: &[u32]) -> Result<Self> {
        let create_info = vk::ShaderModuleCreateInfo::default()
            .code(words);

        let handle = unsafe {
            device.create_shader_module(&create_info, None)
                .context("[VulkanShaderModule] Failed to create raw Shader Module")?
        };

        Ok(Self {
            handle,
            device: device.clone(),
        })
    }

    pub fn from_file<P: AsRef<Path>>(device: &ash::Device, path: P) -> Result<Self> {
        let path_ref = path.as_ref();

        let mut file = File::open(path_ref)
            .with_context(|| format!("[VulkanShaderModule] Failed to open shader file at {:?}", path_ref))?;

        let words = ash::util::read_spv(&mut file)
            .with_context(|| format!("[VulkanShaderModule] Failed to parse SPIR-V from file {:?}", path_ref))?;

        Self::from_words(device, &words)
    }

    pub fn from_bytes(device: &ash::Device, bytes: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(bytes);

        let words = ash::util::read_spv(&mut cursor)
            .context("[VulkanShaderModule] Failed to parse SPIR-V from bytes slice")?;

        Self::from_words(device, &words)
    }
}

impl Drop for VulkanShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.handle, None);
        }
    }
}