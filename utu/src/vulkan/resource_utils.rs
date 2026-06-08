use ash::vk;
use anyhow::{Result, Context};

pub struct VulkanImageView {
    pub handle: vk::ImageView,
    device: ash::Device,
}

impl VulkanImageView {
    pub fn new(
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
    ) -> Result<Self> {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let handle = unsafe {
            device.create_image_view(&view_info, None)
                .context("[VulkanImageView] Failed to create raw ImageView")?
        };

        Ok(Self {
            handle,
            device: device.clone(),
        })
    }
}

impl Drop for VulkanImageView {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.handle, None);
        }
    }
}

pub struct VulkanSampler {
    pub handle: vk::Sampler,
    device: ash::Device,
}

impl VulkanSampler {
    pub fn new(device: &ash::Device, info: &vk::SamplerCreateInfo) -> Result<Self> {
        let handle = unsafe {
            device.create_sampler(info, None)
                .context("[VulkanSampler] Failed to create raw Sampler")?
        };

        Ok(Self {
            handle,
            device: device.clone(),
        })
    }
}

impl Drop for VulkanSampler {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.handle, None);
        }
    }
}

pub fn transition_image_layout(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    aspect_mask: vk::ImageAspectFlags,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    let image_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let barrier = [image_barrier];
    let dependency_info = vk::DependencyInfo::default()
        .image_memory_barriers(&barrier);

    unsafe {
        device.cmd_pipeline_barrier2(cmd, &dependency_info);
    }
}