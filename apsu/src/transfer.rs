use ash::vk;
use anyhow::{Result, Context};
use std::sync::Arc;

use crate::allocator::ApsuAllocator;
use crate::buffer::{GpuDeviceBuffer, GpuUploadBuffer};

pub struct GpuTransferManager {
    allocator_ctx: Arc<ApsuAllocator>,
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,

    staging_buffer: GpuUploadBuffer<u8>,
}

impl GpuTransferManager {
    pub fn new(
        allocator_ctx: Arc<ApsuAllocator>,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        staging_size_in_bytes: usize,
    ) -> Result<Self> {
        let device = allocator_ctx.device.clone();

        let staging_buffer = GpuUploadBuffer::new(
            allocator_ctx.clone(),
            staging_size_in_bytes,
            crate::types::BufferUsage::TRANSFER_SRC,
        ).context("[GpuTransferManager] Failed to allocate persistent staging buffer")?;

        Ok(Self {
            allocator_ctx,
            device,
            queue,
            command_pool,
            staging_buffer,
        })
    }
    pub fn write_buffer<T: bytemuck::Pod>(
        &self,
        dst_buffer: &GpuDeviceBuffer<T>,
        dst_element_offset: usize,
        data: &[T],
    ) -> Result<()> {
        let element_size = size_of::<T>() as u64;
        let data_size_bytes = data.len() as u64 * element_size;
        let dst_offset_bytes = dst_element_offset as u64 * element_size;

        if dst_offset_bytes + data_size_bytes > dst_buffer.size_in_bytes() {
            return Err(anyhow::anyhow!(
                "[GpuTransferManager] Write out of bounds. Buffer size: {}, write attempt at: {} with size: {}",
                dst_buffer.size_in_bytes(),
                dst_offset_bytes,
                data_size_bytes
            ));
        }

        if data_size_bytes <= self.staging_buffer.size_in_bytes() {
            let bytes = bytemuck::cast_slice(data);
            self.staging_buffer.write(bytes)?;

            self.execute_copy_command(
                self.staging_buffer.buffer(),
                dst_buffer.buffer(),
                0,
                dst_offset_bytes,
                data_size_bytes,
            )?;
        } else {
            let temp_staging = GpuUploadBuffer::<T>::new(
                self.allocator_ctx.clone(),
                data.len(),
                crate::types::BufferUsage::TRANSFER_SRC,
            ).context("[GpuTransferManager] Failed to allocate temporary staging buffer")?;

            temp_staging.write(data)?;

            self.execute_copy_command(
                temp_staging.buffer(),
                dst_buffer.buffer(),
                0,
                dst_offset_bytes,
                data_size_bytes,
            )?;
        }

        Ok(())
    }

    fn execute_copy_command(
        &self,
        src_raw: vk::Buffer,
        dst_raw: vk::Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) -> Result<()> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe {
            self.device.allocate_command_buffers(&alloc_info)
                .context("[GpuTransferManager] Failed to allocate temporary command buffer")?[0]
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device.begin_command_buffer(cmd, &begin_info)
                .context("[GpuTransferManager] Failed to begin recording transfer command")?;

            let copy_region = vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(size);

            self.device.cmd_copy_buffer(cmd, src_raw, dst_raw, &[copy_region]);

            self.device.end_command_buffer(cmd)
                .context("[GpuTransferManager] Failed to end recording transfer command")?;
        }

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            self.device.create_fence(&fence_info, None)
                .context("[GpuTransferManager] Failed to create sync fence")?
        };

        let command_buffers = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&command_buffers);

        unsafe {
            self.device.queue_submit(self.queue, &[submit_info], fence)
                .context("[GpuTransferManager] Failed to submit copy command")?;

            self.device.wait_for_fences(&[fence], true, u64::MAX)
                .context("[GpuTransferManager] Failed waiting for transfer fence")?;

            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &[cmd]);
        }

        Ok(())
    }
}