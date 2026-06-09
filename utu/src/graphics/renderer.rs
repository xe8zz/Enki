use ash::vk;
use anyhow::{Result, Context};
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use crate::vulkan::{VulkanSemaphore, transition_image_layout};
use crate::compute::engine::{TimelineCommandRing};
use crate::graphics::window::GpuWindow;

pub struct GpuRenderer {
    pub command_ring: Mutex<TimelineCommandRing>,

    pub timeline_semaphore: VulkanSemaphore,
    pub timeline_counter: AtomicU64,

    device: ash::Device,
    graphics_queue: vk::Queue,
}

impl GpuRenderer {
    pub fn new(
        device: &ash::Device,
        graphics_queue: vk::Queue,
        graphics_family_idx: u32,
        capacity: usize,
    ) -> Result<Self> {
        let command_ring = Mutex::new(
            TimelineCommandRing::new(device, graphics_family_idx, capacity)
                .context("[GpuRenderer] Failed to create graphics command ring")?
        );

        let timeline_semaphore = VulkanSemaphore::new_timeline(device, 0)
            .context("[GpuRenderer] Failed to create graphics timeline semaphore")?;

        let timeline_counter = AtomicU64::new(0);

        Ok(Self {
            command_ring,
            timeline_semaphore,
            timeline_counter,
            device: device.clone(),
            graphics_queue,
        })
    }

    pub fn draw_frame<F>(&self, window: &mut GpuWindow, draw_callback: F) -> Result<()>
    where
        F: FnOnce(vk::CommandBuffer, vk::ImageView, vk::Extent2D) -> Result<()>,
    {
        let (image_index, recreate_swapchain) = window.acquire_next_image(Duration::from_secs(2))?;
        if recreate_swapchain {
            return Err(anyhow::anyhow!("[GpuRenderer] Swapchain is out of date, recreation required"));
        }

        let device = &self.device;

        let mut command_ring = self.command_ring.lock()
            .map_err(|_| anyhow::anyhow!("[GpuRenderer] Failed to lock graphics command ring"))?;

        let slot_idx = command_ring.next_slot_idx % command_ring.slots.len();

        let current_gpu_value = unsafe {
            device.get_semaphore_counter_value(self.timeline_semaphore.handle)
                .context("[GpuRenderer] Failed to query graphics timeline value")?
        };

        let slot = &mut command_ring.slots[slot_idx];

        if current_gpu_value < slot.last_submitted_timeline_value {
            let semaphores = [self.timeline_semaphore.handle];
            let values = [slot.last_submitted_timeline_value];

            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);

            unsafe {
                device.wait_semaphores(&wait_info, u64::MAX)
                    .context("[GpuRenderer] Graphics backpressure wait failed")?;
            }
        }

        unsafe {
            device.reset_command_buffer(slot.cmd, vk::CommandBufferResetFlags::empty())
                .context("[GpuRenderer] Failed to reset graphics command buffer")?;
        }

        let cmd = slot.cmd;
        let next_value = self.timeline_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device.begin_command_buffer(cmd, &begin_info)
                .context("[GpuRenderer] Failed to begin recording graphics command buffer")?;
        }

        let extent = window.extent;
        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(extent);

        unsafe {
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
        }

        let target_image = window.images[image_index as usize];
        let target_view = window.image_views[image_index as usize];

        transition_image_layout(
            device,
            cmd,
            target_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );

        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(target_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.03, 0.03, 0.03, 1.0],
                },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment));

        unsafe {
            device.cmd_begin_rendering(cmd, &rendering_info);
        }

        draw_callback(cmd, target_view, extent)?;

        unsafe {
            device.cmd_end_rendering(cmd);
        }

        transition_image_layout(
            device,
            cmd,
            target_image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageAspectFlags::COLOR,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::NONE,
        );

        unsafe {
            device.end_command_buffer(cmd)
                .context("[GpuRenderer] Failed to finalize graphics command recording")?;
        }

        let wait_semaphores = [window.current_image_acquired_semaphore()];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let signal_semaphores = [
            window.current_render_finished_semaphore(),
            self.timeline_semaphore.handle,
        ];

        let temp_next_value = [0, next_value];
        let mut timeline_submit_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&[0])
            .signal_semaphore_values(&temp_next_value);

        let command_buffers = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .push_next(&mut timeline_submit_info)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            device.queue_submit(self.graphics_queue, &[submit_info], window.current_in_flight_fence())
                .context("[GpuRenderer] Failed to submit graphics commands to queue")?;
        }

        slot.last_submitted_timeline_value = next_value;
        command_ring.next_slot_idx += 1;

        let suboptimal = window.present_image(self.graphics_queue, image_index)
            .context("[GpuRenderer] Failed to present swapchain image")?;

        if suboptimal {
            return Err(anyhow::anyhow!("[GpuRenderer] Swapchain is suboptimal, recreation recommended"));
        }

        Ok(())
    }
}