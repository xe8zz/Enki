use ash::vk;
use anyhow::{Result, Context};
use std::sync::Arc;

use crate::allocator::ApsuAllocator;
use crate::buffer::GpuBuffer;
use crate::types::{MemoryUsage, BufferUsage};

pub struct StagingBuffer<T: bytemuck::Pod> {
    pub buffer: GpuBuffer<T>,
}

impl<T: bytemuck::Pod> StagingBuffer<T> {
    pub fn new(allocator_ctx: Arc<ApsuAllocator>, data: &[T]) -> Result<Self> {
        let buffer = GpuBuffer::new(
            allocator_ctx,
            data.len(),
            BufferUsage::TransferSrc,
            MemoryUsage::HostMappable,
        ).context("[StagingBuffer] Failed to allocate temporary staging GpuBuffer")?;

        buffer.write(data)?;

        Ok(Self { buffer })
    }

    pub fn get_buffer_copy_region(&self, dst_offset_bytes: u64) -> vk::BufferCopy {
        vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(dst_offset_bytes)
            .size(self.buffer.size_in_bytes)
    }

    pub fn get_image_copy_region(
        &self,
        width: u32,
        height: u32,
        aspect_mask: vk::ImageAspectFlags,
    ) -> vk::BufferImageCopy {
        vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
    }
}