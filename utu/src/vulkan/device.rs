use ash::vk;
use anyhow::{Result, anyhow, Context};
use std::ffi::{CStr, CString};

pub struct PhysicalDeviceInfo {
    pub handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub features: vk::PhysicalDeviceFeatures,
    pub queue_families: Vec<vk::QueueFamilyProperties>,
}

impl PhysicalDeviceInfo {
    pub fn query(instance: &ash::Instance, handle: vk::PhysicalDevice) -> Self {
        let properties = unsafe { instance.get_physical_device_properties(handle) };
        let features = unsafe { instance.get_physical_device_features(handle) };
        let queue_families = unsafe {
            instance.get_physical_device_queue_family_properties(handle)
        };

        Self {
            handle,
            properties,
            features,
            queue_families,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VulkanQueue {
    pub handle: vk::Queue,
    pub family_index: u32,
    pub queue_index: u32,
}

pub struct VulkanDevice {
    pub logical_device: ash::Device,
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.logical_device.destroy_device(None);
        }
    }
}

pub struct QueueRequest {
    pub family_index: u32,
    pub priorities: Vec<f32>,
}

pub struct VulkanDeviceBuilder<'a> {
    physical_device: vk::PhysicalDevice,
    required_extensions: Vec<&'a CStr>,
    queue_requests: Vec<QueueRequest>,
    enable_buffer_device_address: bool,
    enable_descriptor_indexing: bool,
}

impl<'a> VulkanDeviceBuilder<'a> {
    pub fn new(physical_device: vk::PhysicalDevice) -> Self {
        Self {
            physical_device,
            required_extensions: Vec::new(),
            queue_requests: Vec::new(),
            enable_buffer_device_address: false,
            enable_descriptor_indexing: false,
        }
    }

    pub fn with_extensions(mut self, extensions: &[&'a CStr]) -> Self {
        self.required_extensions.extend_from_slice(extensions);
        self
    }

    pub fn request_queue(mut self, family_index: u32, priorities: Vec<f32>) -> Self {
        self.queue_requests.push(QueueRequest {
            family_index,
            priorities,
        });
        self
    }

    pub fn enable_buffer_device_address(mut self, enable: bool) -> Self {
        self.enable_buffer_device_address = enable;
        self
    }

    pub fn enable_descriptor_indexing(mut self, enable: bool) -> Self {
        self.enable_descriptor_indexing = enable;
        self
    }

    pub fn build(self, instance: &ash::Instance) -> Result<(VulkanDevice, Vec<VulkanQueue>)> {
        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = self.queue_requests
            .iter()
            .map(|req| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(req.family_index)
                    .queue_priorities(&req.priorities)
            })
            .collect();

        let extension_names: Vec<*const std::os::raw::c_char> = self.required_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect();

        let mut features11 = vk::PhysicalDeviceVulkan11Features::default()
            .shader_draw_parameters(true);

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(self.enable_buffer_device_address)
            .descriptor_indexing(self.enable_descriptor_indexing)
            .runtime_descriptor_array(self.enable_descriptor_indexing)
            .descriptor_binding_partially_bound(self.enable_descriptor_indexing)
            .descriptor_binding_storage_buffer_update_after_bind(self.enable_descriptor_indexing)
            .descriptor_binding_sampled_image_update_after_bind(self.enable_descriptor_indexing)
            .descriptor_binding_storage_image_update_after_bind(self.enable_descriptor_indexing)
            .descriptor_binding_uniform_buffer_update_after_bind(self.enable_descriptor_indexing)
            .timeline_semaphore(true);

        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);


        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut features11)
            .push_next(&mut features12)
            .push_next(&mut features13);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extension_names)
            .push_next(&mut features2);

        let logical_device = unsafe {
            instance.create_device(self.physical_device, &device_create_info, None)
                .context("[VulkanDevice] Failed to create Logical Device")?
        };

        let mut retrieved_queues = Vec::new();
        for req in &self.queue_requests {
            for (queue_idx, _) in req.priorities.iter().enumerate() {
                let queue_handle = unsafe {
                    logical_device.get_device_queue(req.family_index, queue_idx as u32)
                };
                retrieved_queues.push(VulkanQueue {
                    handle: queue_handle,
                    family_index: req.family_index,
                    queue_index: queue_idx as u32,
                });
            }
        }

        Ok((VulkanDevice { logical_device }, retrieved_queues))
    }
}