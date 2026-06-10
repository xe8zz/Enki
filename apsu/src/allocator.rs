use ash::vk;
use vk_mem::Allocator;
use anyhow::{Result, Context};
use std::sync::Arc;

pub struct ApsuAllocator {
    pub allocator: Arc<Allocator>,
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
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


        let memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };

        let mut is_unified_memory = device_properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU;

        if !is_unified_memory {
            let has_unified_heap = memory_properties.memory_heaps[..memory_properties.memory_heap_count as usize]
                .iter()
                .any(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) && heap.size > 0);

            if memory_properties.memory_heap_count == 1 && has_unified_heap {
                is_unified_memory = true;
            }
        }

        let mut create_info = vk_mem::AllocatorCreateInfo::new(
            instance,
            device,
            physical_device
        );

        create_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS
            | vk_mem::AllocatorCreateFlags::EXT_MEMORY_BUDGET;

        let allocator_raw = unsafe {
            Allocator::new(create_info)
                .context("[ApsuAllocator] Failed to create raw VMA Allocator instance")?
        };

        Ok(Self {
            allocator: Arc::new(allocator_raw),
            device: device.clone(),
            physical_device,
            is_unified_memory,
        })
    }
}