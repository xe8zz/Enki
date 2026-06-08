use ash::vk;
use anyhow::{Result, Context};
use std::sync::Arc;
use vk_mem::Alloc;

use crate::allocator::ApsuAllocator;

pub struct GpuImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: vk_mem::Allocation,
    pub format: vk::Format,
    pub extent: vk::Extent3D,

    allocator_ctx: Arc<ApsuAllocator>,
}

impl GpuImage {
    pub fn new(
        allocator_ctx: Arc<ApsuAllocator>,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self> {
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (image, allocation) = unsafe {
            allocator_ctx.allocator.create_image(&image_info, &alloc_info)
                .context("[GpuImage] Failed to allocate VMA Image")?
        };

        let is_depth = format == vk::Format::D16_UNORM
            || format == vk::Format::X8_D24_UNORM_PACK32
            || format == vk::Format::D32_SFLOAT
            || format == vk::Format::S8_UINT
            || format == vk::Format::D16_UNORM_S8_UINT
            || format == vk::Format::D24_UNORM_S8_UINT
            || format == vk::Format::D32_SFLOAT_S8_UINT;

        let aspect_mask = if is_depth {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

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

        let view = unsafe {
            allocator_ctx.device.create_image_view(&view_info, None)
                .context("[GpuImage] Failed to create ImageView for allocated image")?
        };

        Ok(Self {
            image,
            view,
            allocation,
            format,
            extent,
            allocator_ctx,
        })
    }
}

impl Drop for GpuImage {
    fn drop(&mut self) {
        unsafe {
            self.allocator_ctx.device.destroy_image_view(self.view, None);

            self.allocator_ctx.allocator.destroy_image(self.image, &mut self.allocation);
        }
    }
}