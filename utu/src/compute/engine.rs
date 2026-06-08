use ash::vk;
use anyhow::{Result, Context};
use std::sync::Arc;

use crate::vulkan::{
    VulkanInstance, VulkanInstanceBuilder,
    PhysicalDeviceInfo, VulkanQueue, VulkanDevice, VulkanDeviceBuilder,
    VulkanCommandPool,
    VulkanDescriptorSetLayout, DescriptorSetLayoutBuilder,
    VulkanDescriptorPool, DescriptorPoolBuilder,
};

use apsu::{ApsuAllocator, GpuBuffer, BufferUsage, MemoryUsage};

#[derive(Debug, Clone)]
pub struct ComputeEngineConfig {
    pub app_name: String,
    pub max_storage_buffers: u32,
    pub max_uniform_buffers: u32,
}

impl Default for ComputeEngineConfig {
    fn default() -> Self {
        Self {
            app_name: "Enki Compute Engine".to_string(),
            max_storage_buffers: 1024,
            max_uniform_buffers: 256,
        }
    }
}

pub struct ComputeEngine {
    pub instance: VulkanInstance,
    pub device: VulkanDevice,
    pub queue: VulkanQueue,
    pub command_pool: VulkanCommandPool,

    pub allocator: Arc<ApsuAllocator>,

    pub descriptor_layout: VulkanDescriptorSetLayout,
    pub descriptor_pool: VulkanDescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
}

impl ComputeEngine {
    pub fn new(config: ComputeEngineConfig) -> Result<Self> {
        let instance = VulkanInstanceBuilder::new(config.app_name.as_str())
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

        let compute_family_idx = selected_gpu_info.queue_families.iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .unwrap() as u32;

        let (device, mut queues) = VulkanDeviceBuilder::new(selected_gpu_info.handle)
            .request_queue(compute_family_idx, vec![1.0])
            .enable_buffer_device_address(true)
            .enable_descriptor_indexing(true)
            .build(&instance.instance)
            .context("[ComputeEngine] Failed to build Vulkan Logical Device")?;

        let queue = queues.remove(0);

        let allocator = Arc::new(ApsuAllocator::new(
            &instance.instance,
            selected_gpu_info.handle,
            &device.logical_device,
        ).context("[ComputeEngine] Failed to initialize Apsu Allocator")?);

        let command_pool = VulkanCommandPool::new(
            &device.logical_device,
            compute_family_idx,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        ).context("[ComputeEngine] Failed to create Compute Command Pool")?;

        let descriptor_layout = DescriptorSetLayoutBuilder::new()
            .add_binding(
                0,
                vk::DescriptorType::STORAGE_BUFFER,
                config.max_storage_buffers,
                vk::ShaderStageFlags::COMPUTE,
                vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            )
            .add_binding(
                1,
                vk::DescriptorType::UNIFORM_BUFFER,
                config.max_uniform_buffers,
                vk::ShaderStageFlags::COMPUTE,
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
            command_pool,
            allocator,
            descriptor_layout,
            descriptor_pool,
            descriptor_set,
        })
    }

    pub fn create_buffer<T: bytemuck::Pod>(
        &self,
        element_count: usize,
        usage: BufferUsage,
        memory_usage: MemoryUsage,
    ) -> Result<GpuBuffer<T>> {
        GpuBuffer::new(self.allocator.clone(), element_count, usage, memory_usage)
    }

    pub fn bind_storage_buffer<T: bytemuck::Pod>(&self, slot_index: u32, buffer: &GpuBuffer<T>) -> Result<()> {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer.buffer)
            .offset(0)
            .range(buffer.size_in_bytes);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .dst_array_element(slot_index)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));

        unsafe {
            self.device.logical_device.update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }
        Ok(())
    }

    pub fn bind_uniform_buffer<T: bytemuck::Pod>(&self, slot_index: u32, buffer: &GpuBuffer<T>) -> Result<()> {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer.buffer)
            .offset(0)
            .range(buffer.size_in_bytes);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(1)
            .dst_array_element(slot_index)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));

        unsafe {
            self.device.logical_device.update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }
        Ok(())
    }
}