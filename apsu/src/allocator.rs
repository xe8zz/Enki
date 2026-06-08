use ash::vk;
use vk_mem::Allocator;
use anyhow::{Result, Context};
use std::sync::Arc;

pub struct ApsuAllocator {
    pub allocator: Arc<Allocator>,
    pub device: ash::Device,
    pub is_unified_memory: bool,
}

impl ApsuAllocator {
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
    ) -> Result<Self> {
        let device_properties = unsafe {
            instance.get_physical_device_properties(physical_device)
        };

        let is_unified_memory = device_properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU;

        let mut create_info = vk_mem::AllocatorCreateInfo::new(
            instance,
            device,
            physical_device
        );

        create_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

        let allocator_raw = unsafe {
            Allocator::new(create_info)
                .context("[ApsuAllocator] Failed to create raw VMA Allocator instance")?
        };

        Ok(Self {
            allocator: Arc::new(allocator_raw),
            device: device.clone(),
            is_unified_memory,
        })
    }
}