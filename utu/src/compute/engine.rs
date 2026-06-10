use ash::vk;
use anyhow::{Result, Context};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::ffi::CStr;

use crate::vulkan::{
    VulkanInstance, VulkanInstanceBuilder,
    PhysicalDeviceInfo, VulkanQueue, VulkanDevice, VulkanDeviceBuilder,
    VulkanCommandPool, VulkanSemaphore,
    VulkanDescriptorSetLayout, DescriptorSetLayoutBuilder,
    VulkanDescriptorPool, DescriptorPoolBuilder,
};

use apsu::{
    ApsuAllocator, GpuDeviceBuffer, GpuSharedBuffer, GpuUploadBuffer, GpuReadbackBuffer
};

pub struct PendingDescriptorWrite {
    pub slot_index: u32,
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub buffer: vk::Buffer,
    pub range: u64,
}

pub struct CommandSlot {
    pub cmd: vk::CommandBuffer,
    pub last_submitted_timeline_value: u64,
}

pub struct TimelineCommandRing {
    pub command_pool: VulkanCommandPool,
    pub slots: Vec<CommandSlot>,
    pub next_slot_idx: usize,
}

impl TimelineCommandRing {
    pub fn new(device: &ash::Device, queue_family_index: u32, capacity: usize) -> Result<Self> {
        let command_pool = VulkanCommandPool::new(
            device,
            queue_family_index,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        ).context("[TimelineCommandRing] Failed to create command pool")?;

        let raw_buffers = command_pool.allocate_buffers(
            vk::CommandBufferLevel::PRIMARY,
            capacity as u32,
        ).context("[TimelineCommandRing] Failed to allocate initial command buffers")?;

        let slots = raw_buffers.into_iter()
            .map(|cmd| CommandSlot {
                cmd,
                last_submitted_timeline_value: 0,
            })
            .collect();

        Ok(Self {
            command_pool,
            slots,
            next_slot_idx: 0,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ComputeEngineConfig {
    pub app_name: String,
    pub max_storage_buffers: u32,
    pub max_uniform_buffers: u32,

    pub required_instance_extensions: Vec<std::ffi::CString>,
    pub required_device_extensions: Vec<std::ffi::CString>,
    pub request_graphics_queue: bool,
}

impl Default for ComputeEngineConfig {
    fn default() -> Self {
        Self {
            app_name: "Enki Compute Engine".to_string(),
            max_storage_buffers: 1024,
            max_uniform_buffers: 256,
            required_instance_extensions: Vec::new(),
            required_device_extensions: Vec::new(),
            request_graphics_queue: false,
        }
    }
}

pub struct ComputeEngine {
    pub instance: VulkanInstance,
    pub device: VulkanDevice,
    pub queue: VulkanQueue,

    pub command_ring: Mutex<TimelineCommandRing>,
    pub timeline_semaphore: VulkanSemaphore,
    pub timeline_counter: AtomicU64,

    pub pending_writes: Mutex<Vec<PendingDescriptorWrite>>,

    pub allocator: Arc<ApsuAllocator>,

    pub descriptor_layout: VulkanDescriptorSetLayout,
    pub descriptor_pool: VulkanDescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
}

impl ComputeEngine {
    pub fn new(config: ComputeEngineConfig) -> Result<Self> {
        let required_instance_extensions: Vec<&CStr> = config.required_instance_extensions
            .iter()
            .map(|c| c.as_c_str())
            .collect();

        let instance = VulkanInstanceBuilder::new(config.app_name.as_str())
            .with_extensions(&required_instance_extensions)
            .build()
            .context("[ComputeEngine] Failed to build Vulkan Instance")?;

        let physical_devices = unsafe {
            instance.instance.enumerate_physical_devices()
                .context("[ComputeEngine] Failed to enumerate physical devices")?
        };

        let devices_info: Vec<PhysicalDeviceInfo> = physical_devices.into_iter()
            .map(|pd| PhysicalDeviceInfo::query(&instance.instance, pd))
            .collect();

        let selected_gpu_info = devices_info.iter()
            .find(|info| {
                info.properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU &&
                    info.queue_families.iter().any(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            })
            .or_else(|| {
                devices_info.iter().find(|info| {
                    info.queue_families.iter().any(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                })
            })
            .ok_or_else(|| anyhow::anyhow!("[ComputeEngine] No physical device with GPGPU/Compute support found"))?;

        let queue_family_idx = if config.request_graphics_queue {
            selected_gpu_info.queue_families.iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS))
                .or_else(|| {
                    selected_gpu_info.queue_families.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                })
                .ok_or_else(|| anyhow::anyhow!("[ComputeEngine] No queue family with Compute + Graphics support found"))? as u32
        } else {
            selected_gpu_info.queue_families.iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or_else(|| anyhow::anyhow!("[ComputeEngine] No queue family with Compute support found"))? as u32
        };

        let required_device_extensions: Vec<&CStr> = config.required_device_extensions
            .iter()
            .map(|c| c.as_c_str())
            .collect();

        let (device, mut queues) = VulkanDeviceBuilder::new(selected_gpu_info.handle)
            .request_queue(queue_family_idx, vec![1.0])
            .enable_buffer_device_address(true)
            .enable_descriptor_indexing(true)
            .with_extensions(&required_device_extensions)
            .build(&instance.instance)
            .context("[ComputeEngine] Failed to build Vulkan Logical Device")?;

        let queue = queues.remove(0);

        let allocator = Arc::new(ApsuAllocator::new(
            &instance.instance,
            selected_gpu_info.handle,
            &device.logical_device,
        ).context("[ComputeEngine] Failed to initialize Apsu Allocator")?);

        let command_ring = Mutex::new(
            TimelineCommandRing::new(&device.logical_device, queue_family_idx, 4)
                .context("[ComputeEngine] Failed to initialize Timeline Command Ring")?
        );

        let timeline_semaphore = VulkanSemaphore::new_timeline(&device.logical_device, 0)
            .context("[ComputeEngine] Failed to create Timeline Semaphore")?;

        let timeline_counter = AtomicU64::new(0);
        let pending_writes = Mutex::new(Vec::new());

        let descriptor_layout = DescriptorSetLayoutBuilder::new()
            .add_binding(
                0,
                vk::DescriptorType::STORAGE_BUFFER,
                config.max_storage_buffers,
                vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            )
            .add_binding(
                1,
                vk::DescriptorType::UNIFORM_BUFFER,
                config.max_uniform_buffers,
                vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            )
            .build(&device.logical_device)
            .context("[ComputeEngine] Failed to build Bindless Descriptor Set Layout")?;

        let descriptor_pool = DescriptorPoolBuilder::new()
            .add_pool_size(vk::DescriptorType::STORAGE_BUFFER, config.max_storage_buffers)
            .add_pool_size(vk::DescriptorType::UNIFORM_BUFFER, config.max_uniform_buffers)
            .with_max_sets(1)
            .with_flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .build(&device.logical_device)
            .context("[ComputeEngine] Failed to build Bindless Descriptor Pool")?;

        let descriptor_set = descriptor_pool.allocate_set(descriptor_layout.handle)
            .context("[ComputeEngine] Failed to allocate Bindless Descriptor Set")?;

        Ok(Self {
            instance,
            device,
            queue,
            command_ring,
            timeline_semaphore,
            timeline_counter,
            pending_writes,
            allocator,
            descriptor_layout,
            descriptor_pool,
            descriptor_set,
        })
    }

    pub fn flush_descriptor_updates(&self) -> Result<()> {
        let mut writes = self.pending_writes.lock()
            .map_err(|_| anyhow::anyhow!("[ComputeEngine] Failed to lock pending writes"))?;

        if writes.is_empty() {
            return Ok(());
        }

        let buffer_infos: Vec<vk::DescriptorBufferInfo> = writes
            .iter()
            .map(|w| {
                vk::DescriptorBufferInfo::default()
                    .buffer(w.buffer)
                    .offset(0)
                    .range(w.range)
            })
            .collect();

        let write_sets: Vec<vk::WriteDescriptorSet> = writes
            .iter()
            .enumerate()
            .map(|(i, w)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(w.binding)
                    .dst_array_element(w.slot_index)
                    .descriptor_type(w.descriptor_type)
                    .buffer_info(std::slice::from_ref(&buffer_infos[i]))
            })
            .collect();

        unsafe {
            self.device.logical_device.update_descriptor_sets(&write_sets, &[]);
        }

        writes.clear();
        Ok(())
    }

    pub fn create_device_buffer<T: bytemuck::Pod>(&self, element_count: usize, usage: apsu::BufferUsage) -> Result<GpuDeviceBuffer<T>> {
        GpuDeviceBuffer::new(self.allocator.clone(), element_count, usage)
    }

    pub fn create_shared_buffer<T: bytemuck::Pod>(&self, element_count: usize, usage: apsu::BufferUsage) -> Result<GpuSharedBuffer<T>> {
        GpuSharedBuffer::new(self.allocator.clone(), element_count, usage)
    }

    pub fn create_upload_buffer<T: bytemuck::Pod>(&self, element_count: usize, usage: apsu::BufferUsage) -> Result<GpuUploadBuffer<T>> {
        GpuUploadBuffer::new(self.allocator.clone(), element_count, usage)
    }

    pub fn create_readback_buffer<T: bytemuck::Pod>(&self, element_count: usize, usage: apsu::BufferUsage) -> Result<GpuReadbackBuffer<T>> {
        GpuReadbackBuffer::new(self.allocator.clone(), element_count, usage)
    }

    fn bind_storage_buffer_raw(&self, slot_index: u32, buffer: vk::Buffer, size_in_bytes: u64) -> Result<()> {
        let mut writes = self.pending_writes.lock()
            .map_err(|_| anyhow::anyhow!("[ComputeEngine] Failed to lock pending writes"))?;

        writes.push(PendingDescriptorWrite {
            slot_index,
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            buffer,
            range: size_in_bytes,
        });
        Ok(())
    }

    fn bind_uniform_buffer_raw(&self, slot_index: u32, buffer: vk::Buffer, size_in_bytes: u64) -> Result<()> {
        let mut writes = self.pending_writes.lock()
            .map_err(|_| anyhow::anyhow!("[ComputeEngine] Failed to lock pending writes"))?;

        writes.push(PendingDescriptorWrite {
            slot_index,
            binding: 1,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            buffer,
            range: size_in_bytes,
        });
        Ok(())
    }

    pub fn bind_storage_device_buffer<T: bytemuck::Pod>(&self, slot_index: u32, buffer: &GpuDeviceBuffer<T>) -> Result<()> {
        self.bind_storage_buffer_raw(slot_index, buffer.buffer(), buffer.size_in_bytes())
    }

    pub fn bind_storage_shared_buffer<T: bytemuck::Pod>(&self, slot_index: u32, buffer: &GpuSharedBuffer<T>) -> Result<()> {
        self.bind_storage_buffer_raw(slot_index, buffer.buffer(), buffer.size_in_bytes())
    }

    pub fn bind_storage_upload_buffer<T: bytemuck::Pod>(&self, slot_index: u32, buffer: &GpuUploadBuffer<T>) -> Result<()> {
        self.bind_storage_buffer_raw(slot_index, buffer.buffer(), buffer.size_in_bytes())
    }

    pub fn bind_storage_readback_buffer<T: bytemuck::Pod>(&self, slot_index: u32, buffer: &GpuReadbackBuffer<T>) -> Result<()> {
        self.bind_storage_buffer_raw(slot_index, buffer.buffer(), buffer.size_in_bytes())
    }

    pub fn bind_uniform_device_buffer<T: bytemuck::Pod>(&self, slot_index: u32, buffer: &GpuDeviceBuffer<T>) -> Result<()> {
        self.bind_uniform_buffer_raw(slot_index, buffer.buffer(), buffer.size_in_bytes())
    }

    pub fn bind_uniform_shared_buffer<T: bytemuck::Pod>(&self, slot_index: u32, buffer: &GpuSharedBuffer<T>) -> Result<()> {
        self.bind_uniform_buffer_raw(slot_index, buffer.buffer(), buffer.size_in_bytes())
    }

    pub fn bind_uniform_upload_buffer<T: bytemuck::Pod>(&self, slot_index: u32, buffer: &GpuUploadBuffer<T>) -> Result<()> {
        self.bind_uniform_buffer_raw(slot_index, buffer.buffer(), buffer.size_in_bytes())
    }

    pub fn bind_uniform_readback_buffer<T: bytemuck::Pod>(&self, slot_index: u32, buffer: &GpuReadbackBuffer<T>) -> Result<()> {
        self.bind_uniform_buffer_raw(slot_index, buffer.buffer(), buffer.size_in_bytes())
    }
}