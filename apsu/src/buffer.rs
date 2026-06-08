use ash::vk;
use anyhow::{Result, Context};
use std::sync::Arc;
use std::marker::PhantomData;
use vk_mem::Alloc;

use crate::allocator::ApsuAllocator;
use crate::types::{MemoryUsage, BufferUsage};

pub struct GpuBuffer<T: bytemuck::Pod> {
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    pub device_address: u64,
    pub size_in_bytes: u64,
    pub element_count: usize,

    allocator_ctx: Arc<ApsuAllocator>,
    _phantom: PhantomData<T>,
}

impl<T: bytemuck::Pod> GpuBuffer<T> {
    pub fn new(
        allocator_ctx: Arc<ApsuAllocator>,
        element_count: usize,
        usage: BufferUsage,
        memory_usage: MemoryUsage,
    ) -> Result<Self> {
        let element_size = size_of::<T>() as u64;
        let size_in_bytes = element_count as u64 * element_size;

        let vk_usage = usage.to_vk()
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size_in_bytes)
            .usage(vk_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let (vma_usage, vma_flags) = memory_usage.to_vma(allocator_ctx.is_unified_memory);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vma_usage,
            flags: vma_flags,
            ..Default::default()
        };

        let (buffer, allocation) = unsafe {
            allocator_ctx.allocator.create_buffer(&buffer_info, &alloc_info)
                .context("[GpuBuffer] Failed to allocate VMA Buffer")?
        };

        let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let device_address = unsafe {
            allocator_ctx.device.get_buffer_device_address(&address_info)
        };

        Ok(Self {
            buffer,
            allocation,
            device_address,
            size_in_bytes,
            element_count,
            allocator_ctx,
            _phantom: PhantomData,
        })
    }

    pub fn write(&self, data: &[T]) -> Result<()> {
        let alloc_info = self.allocator_ctx.allocator.get_allocation_info(&self.allocation);

        if alloc_info.mapped_data.is_null() {
            return Err(anyhow::anyhow!(
                "[GpuBuffer] Cannot write directly to non-mappable (DeviceLocal) buffer. Use staging pipeline."
            ));
        }

        let bytes_to_write = data.len() as u64 * size_of::<T>() as u64;
        if bytes_to_write > self.size_in_bytes {
            return Err(anyhow::anyhow!("[GpuBuffer] Input data size exceeds allocated Buffer size"));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                alloc_info.mapped_data as *mut T,
                data.len()
            );
        }

        Ok(())
    }

    pub fn read(&self, out_data: &mut [T]) -> Result<()> {
        let alloc_info = self.allocator_ctx.allocator.get_allocation_info(&self.allocation);

        if alloc_info.mapped_data.is_null() {
            return Err(anyhow::anyhow!(
                "[GpuBuffer] Cannot read directly from non-mappable (DeviceLocal) buffer."
            ));
        }

        if out_data.len() > self.element_count {
            return Err(anyhow::anyhow!("[GpuBuffer] Output slice is too small to receive buffer elements"));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                alloc_info.mapped_data as *const T,
                out_data.as_mut_ptr(),
                out_data.len()
            );
        }

        Ok(())
    }
}

impl<T: bytemuck::Pod> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            self.allocator_ctx.allocator.destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}