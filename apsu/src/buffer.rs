use ash::vk;
use anyhow::{Result, Context};
use std::sync::Arc;
use std::marker::PhantomData;
use vk_mem::Alloc;

use crate::allocator::ApsuAllocator;
use crate::types::{MemoryUsage, BufferUsage};

struct RawGpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    pub device_address: u64,
    pub size_in_bytes: u64,
    pub mapped_ptr: *mut std::ffi::c_void,
    allocator_ctx: Arc<ApsuAllocator>,
}

impl RawGpuBuffer {
    fn new(
        allocator_ctx: Arc<ApsuAllocator>,
        size_in_bytes: u64,
        usage: BufferUsage,
        memory_usage: MemoryUsage,
    ) -> Result<Self> {
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
                .context("[RawGpuBuffer] Failed to allocate VMA Buffer")?
        };

        let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let device_address = unsafe {
            allocator_ctx.device.get_buffer_device_address(&address_info)
        };

        let mut mapped_ptr = std::ptr::null_mut();
        if vma_flags.contains(vk_mem::AllocationCreateFlags::MAPPED) {
            let alloc_info = allocator_ctx.allocator.get_allocation_info(&allocation);
            mapped_ptr = alloc_info.mapped_data;
        }

        Ok(Self {
            buffer,
            allocation,
            device_address,
            size_in_bytes,
            mapped_ptr,
            allocator_ctx,
        })
    }
}

impl Drop for RawGpuBuffer {
    fn drop(&mut self) {
        unsafe {
            self.allocator_ctx.allocator.destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}

pub struct GpuDeviceBuffer<T: bytemuck::Pod> {
    raw: RawGpuBuffer,
    pub element_count: usize,
    _phantom: PhantomData<T>,
}

impl<T: bytemuck::Pod> GpuDeviceBuffer<T> {
    pub fn new(
        allocator_ctx: Arc<ApsuAllocator>,
        element_count: usize,
        usage: BufferUsage,
    ) -> Result<Self> {
        let size_in_bytes = element_count as u64 * size_of::<T>() as u64;
        let raw = RawGpuBuffer::new(allocator_ctx, size_in_bytes, usage, MemoryUsage::DeviceOnly)?;

        Ok(Self {
            raw,
            element_count,
            _phantom: PhantomData,
        })
    }

    pub fn buffer(&self) -> vk::Buffer { self.raw.buffer }
    pub fn device_address(&self) -> u64 { self.raw.device_address }
    pub fn size_in_bytes(&self) -> u64 { self.raw.size_in_bytes }
}

unsafe impl<T: bytemuck::Pod> Send for GpuDeviceBuffer<T> {}
unsafe impl<T: bytemuck::Pod> Sync for GpuDeviceBuffer<T> {}

pub struct GpuUploadBuffer<T: bytemuck::Pod> {
    raw: RawGpuBuffer,
    pub element_count: usize,
    _phantom: PhantomData<T>,
}

impl<T: bytemuck::Pod> GpuUploadBuffer<T> {
    pub fn new(
        allocator_ctx: Arc<ApsuAllocator>,
        element_count: usize,
        usage: BufferUsage,
    ) -> Result<Self> {
        let size_in_bytes = element_count as u64 * size_of::<T>() as u64;
        let raw = RawGpuBuffer::new(allocator_ctx, size_in_bytes, usage, MemoryUsage::Upload)?;

        Ok(Self {
            raw,
            element_count,
            _phantom: PhantomData,
        })
    }

    pub fn write(&self, data: &[T]) -> Result<()> {
        if self.raw.mapped_ptr.is_null() {
            return Err(anyhow::anyhow!("[GpuUploadBuffer] Mapping failed during allocation"));
        }

        let bytes_to_write = data.len() as u64 * size_of::<T>() as u64;
        if bytes_to_write > self.raw.size_in_bytes {
            return Err(anyhow::anyhow!("[GpuUploadBuffer] Input data exceeds buffer capacity"));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.raw.mapped_ptr as *mut T,
                data.len(),
            );
        }
        Ok(())
    }

    pub fn buffer(&self) -> vk::Buffer { self.raw.buffer }
    pub fn device_address(&self) -> u64 { self.raw.device_address }
    pub fn size_in_bytes(&self) -> u64 { self.raw.size_in_bytes }
}

unsafe impl<T: bytemuck::Pod> Send for GpuUploadBuffer<T> {}
unsafe impl<T: bytemuck::Pod> Sync for GpuUploadBuffer<T> {}

pub struct GpuReadbackBuffer<T: bytemuck::Pod> {
    raw: RawGpuBuffer,
    pub element_count: usize,
    _phantom: PhantomData<T>,
}

impl<T: bytemuck::Pod> GpuReadbackBuffer<T> {
    pub fn new(
        allocator_ctx: Arc<ApsuAllocator>,
        element_count: usize,
        usage: BufferUsage,
    ) -> Result<Self> {
        let size_in_bytes = element_count as u64 * size_of::<T>() as u64;
        let raw = RawGpuBuffer::new(allocator_ctx, size_in_bytes, usage, MemoryUsage::Download)?;

        Ok(Self {
            raw,
            element_count,
            _phantom: PhantomData,
        })
    }

    pub fn read(&self, out_data: &mut [T]) -> Result<()> {
        if self.raw.mapped_ptr.is_null() {
            return Err(anyhow::anyhow!("[GpuReadbackBuffer] Mapping failed during allocation"));
        }

        if out_data.len() > self.element_count {
            return Err(anyhow::anyhow!("[GpuReadbackBuffer] Output slice is too small"));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                self.raw.mapped_ptr as *const T,
                out_data.as_mut_ptr(),
                out_data.len(),
            );
        }
        Ok(())
    }

    pub fn buffer(&self) -> vk::Buffer { self.raw.buffer }
    pub fn device_address(&self) -> u64 { self.raw.device_address }
    pub fn size_in_bytes(&self) -> u64 { self.raw.size_in_bytes }
}

unsafe impl<T: bytemuck::Pod> Send for GpuReadbackBuffer<T> {}
unsafe impl<T: bytemuck::Pod> Sync for GpuReadbackBuffer<T> {}

pub struct GpuSharedBuffer<T: bytemuck::Pod> {
    raw: RawGpuBuffer,
    pub element_count: usize,
    _phantom: PhantomData<T>,
}

impl<T: bytemuck::Pod> GpuSharedBuffer<T> {
    pub fn new(
        allocator_ctx: Arc<ApsuAllocator>,
        element_count: usize,
        usage: BufferUsage,
    ) -> Result<Self> {
        let size_in_bytes = element_count as u64 * size_of::<T>() as u64;
        let raw = RawGpuBuffer::new(allocator_ctx, size_in_bytes, usage, MemoryUsage::ZeroCopy)?;

        Ok(Self {
            raw,
            element_count,
            _phantom: PhantomData,
        })
    }

    pub fn write(&self, data: &[T]) -> Result<()> {
        if self.raw.mapped_ptr.is_null() {
            return Err(anyhow::anyhow!("[GpuSharedBuffer] Mapping failed during allocation"));
        }

        let bytes_to_write = data.len() as u64 * size_of::<T>() as u64;
        if bytes_to_write > self.raw.size_in_bytes {
            return Err(anyhow::anyhow!("[GpuSharedBuffer] Input data exceeds buffer capacity"));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.raw.mapped_ptr as *mut T,
                data.len(),
            );
        }
        Ok(())
    }

    pub fn read(&self, out_data: &mut [T]) -> Result<()> {
        if self.raw.mapped_ptr.is_null() {
            return Err(anyhow::anyhow!("[GpuSharedBuffer] Mapping failed during allocation"));
        }

        if out_data.len() > self.element_count {
            return Err(anyhow::anyhow!("[GpuSharedBuffer] Output slice is too small"));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                self.raw.mapped_ptr as *const T,
                out_data.as_mut_ptr(),
                out_data.len(),
            );
        }
        Ok(())
    }

    pub fn buffer(&self) -> vk::Buffer { self.raw.buffer }
    pub fn device_address(&self) -> u64 { self.raw.device_address }
    pub fn size_in_bytes(&self) -> u64 { self.raw.size_in_bytes }
}

unsafe impl<T: bytemuck::Pod> Send for GpuSharedBuffer<T> {}
unsafe impl<T: bytemuck::Pod> Sync for GpuSharedBuffer<T> {}