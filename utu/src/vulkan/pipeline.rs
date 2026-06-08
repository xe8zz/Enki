use ash::vk;
use anyhow::{Result, Context};
use std::ffi::CStr;

pub struct VulkanPipelineLayout {
    pub handle: vk::PipelineLayout,
    device: ash::Device,
}

impl VulkanPipelineLayout {
    pub fn new(
        device: &ash::Device,
        set_layouts: &[vk::DescriptorSetLayout],
        push_constants: &[vk::PushConstantRange],
    ) -> Result<Self> {
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(set_layouts)
            .push_constant_ranges(push_constants);

        let handle = unsafe {
            device.create_pipeline_layout(&layout_info, None)
                .context("[VulkanPipelineLayout] Failed to create raw Pipeline Layout")?
        };

        Ok(Self {
            handle,
            device: device.clone(),
        })
    }
}

impl Drop for VulkanPipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_layout(self.handle, None);
        }
    }
}

pub struct VulkanPipeline {
    pub handle: vk::Pipeline,
    device: ash::Device,
}

impl VulkanPipeline {
    pub fn new_compute(
        device: &ash::Device,
        layout: vk::PipelineLayout,
        shader_module: vk::ShaderModule,
        entry_point: &CStr,
    ) -> Result<Self> {
        let shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(shader_stage)
            .layout(layout);

        let pipelines = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| anyhow::anyhow!("[VulkanPipeline] Compute Pipeline Creation Failed: {}", e))?
        };

        Ok(Self {
            handle: pipelines[0],
            device: device.clone(),
        })
    }

    pub fn new_graphics(
        device: &ash::Device,
        layout: vk::PipelineLayout,
        shader_stages: &[vk::PipelineShaderStageCreateInfo],
        input_assembly: &vk::PipelineInputAssemblyStateCreateInfo,
        rasterization: &vk::PipelineRasterizationStateCreateInfo,
        multisample: &vk::PipelineMultisampleStateCreateInfo,
        depth_stencil: &vk::PipelineDepthStencilStateCreateInfo,
        color_blend: &vk::PipelineColorBlendStateCreateInfo,
        dynamic_state: &vk::PipelineDynamicStateCreateInfo,
        color_formats: &[vk::Format],
        depth_format: vk::Format,
    ) -> Result<Self> {
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(color_formats)
            .depth_attachment_format(depth_format);

        let viewport_info = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut rendering_info)
            .stages(shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(input_assembly)
            .viewport_state(&viewport_info)
            .rasterization_state(rasterization)
            .multisample_state(multisample)
            .depth_stencil_state(depth_stencil)
            .color_blend_state(color_blend)
            .dynamic_state(dynamic_state)
            .layout(layout);

        let pipelines = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| anyhow::anyhow!("[VulkanPipeline] Graphics Pipeline Creation Failed: {}", e))?
        };

        Ok(Self {
            handle: pipelines[0],
            device: device.clone(),
        })
    }
}

impl Drop for VulkanPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.handle, None);
        }
    }
}