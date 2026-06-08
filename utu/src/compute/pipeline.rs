use ash::vk;
use anyhow::{Result, Context};
use std::ffi::CString;

use crate::vulkan::{VulkanPipeline, VulkanPipelineLayout, VulkanShaderModule};
use crate::compute::engine::ComputeEngine;

pub struct ComputePipeline {
    pub pipeline: VulkanPipeline,
    pub layout: VulkanPipelineLayout,
}

impl ComputePipeline {
    pub fn new(
        engine: &ComputeEngine,
        spirv_code: &[u8],
    ) -> Result<Self> {
        Self::new_with_config(engine, spirv_code, "main", 0)
    }

    pub fn new_with_config(
        engine: &ComputeEngine,
        spirv_code: &[u8],
        entry_point: &str,
        push_constants_size: u32,
    ) -> Result<Self> {
        let logical_device = &engine.device.logical_device;

        let shader_module = VulkanShaderModule::from_bytes(logical_device, spirv_code)
            .context("[ComputePipeline] Failed to create Shader Module from SPIR-V bytes")?;

        let mut push_constants = Vec::new();
        if push_constants_size > 0 {
            push_constants.push(vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: push_constants_size,
            });
        }

        let set_layouts = [engine.descriptor_layout.handle];
        let layout = VulkanPipelineLayout::new(
            logical_device,
            &set_layouts,
            &push_constants,
        ).context("[ComputePipeline] Failed to create Pipeline Layout")?;

        let entry_point_cstr = CString::new(entry_point)
            .context("[ComputePipeline] Failed to parse entry point CString")?;

        let pipeline = VulkanPipeline::new_compute(
            logical_device,
            layout.handle,
            shader_module.handle,
            &entry_point_cstr,
        ).context("[ComputePipeline] Failed to compile raw Compute Pipeline")?;

        Ok(Self {
            pipeline,
            layout,
        })
    }
}