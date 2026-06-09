use ash::vk;
use anyhow::{Result, Context};
use std::time::Duration;

use crate::compute::engine::ComputeEngine;
use crate::compute::pipeline::ComputePipeline;

pub struct ComputeExecutionTask {
    timeline_semaphore: vk::Semaphore,
    target_value: u64,
    device: ash::Device,
}

impl ComputeExecutionTask {
    pub fn is_completed(&self) -> Result<bool> {
        let current_value = unsafe {
            self.device.get_semaphore_counter_value(self.timeline_semaphore)
                .context("[ComputeExecutionTask] Failed to query timeline semaphore counter")?
        };
        Ok(current_value >= self.target_value)
    }

    pub fn wait(&self, timeout: Duration) -> Result<()> {
        let semaphores = [self.timeline_semaphore];
        let values = [self.target_value];

        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);

        unsafe {
            self.device.wait_semaphores(&wait_info, timeout.as_nanos() as u64)
                .context("[ComputeExecutionTask] Failed waiting for timeline semaphore")?;
        }
        Ok(())
    }

    pub fn target_value(&self) -> u64 {
        self.target_value
    }
}

pub struct ComputeExecutor<'a> {
    engine: &'a ComputeEngine,
}

impl<'a> ComputeExecutor<'a> {
    pub fn new(engine: &'a ComputeEngine) -> Self {
        Self { engine }
    }

    pub fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        grid_size: (u32, u32, u32),
        push_constants: &[u8],
        insert_barrier: bool,
    ) -> Result<()> {
        let task = self.dispatch_async(pipeline, grid_size, push_constants, insert_barrier)?;
        task.wait(Duration::from_secs(3600))?;
        Ok(())
    }

    pub fn dispatch_async(
        &self,
        pipeline: &ComputePipeline,
        grid_size: (u32, u32, u32),
        push_constants: &[u8],
        insert_barrier: bool,
    ) -> Result<ComputeExecutionTask> {
        let device = &self.engine.device.logical_device;

        self.engine.flush_descriptor_updates()
            .context("[ComputeExecutor] Failed to flush pending descriptor updates")?;

        let mut command_ring = self.engine.command_ring.lock()
            .map_err(|_| anyhow::anyhow!("[ComputeExecutor] Failed to lock Command Ring Mutex"))?;

        let slot_idx = command_ring.next_slot_idx % command_ring.slots.len();

        let current_gpu_value = unsafe {
            device.get_semaphore_counter_value(self.engine.timeline_semaphore.handle)
                .context("[ComputeExecutor] Failed to query GPU timeline semaphore value")?
        };

        let slot = &mut command_ring.slots[slot_idx];

        if current_gpu_value < slot.last_submitted_timeline_value {
            let semaphores = [self.engine.timeline_semaphore.handle];
            let values = [slot.last_submitted_timeline_value];

            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);

            unsafe {
                device.wait_semaphores(&wait_info, u64::MAX)
                    .context("[ComputeExecutor] Backpressure wait failed")?;
            }
        }

        unsafe {
            device.reset_command_buffer(slot.cmd, vk::CommandBufferResetFlags::empty())
                .context("[ComputeExecutor] Failed to reset command buffer")?;
        }

        let cmd = slot.cmd;

        let next_value = self.engine.timeline_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device.begin_command_buffer(cmd, &begin_info)
                .context("[ComputeExecutor] Failed to start recording compute commands")?;
        }

        if insert_barrier {
            let memory_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);

            let dependency_info = vk::DependencyInfo::default()
                .memory_barriers(std::slice::from_ref(&memory_barrier));

            unsafe {
                device.cmd_pipeline_barrier2(cmd, &dependency_info);
            }
        }

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline.handle);

            let set = [self.engine.descriptor_set];
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout.handle,
                0,
                &set,
                &[],
            );
        }

        if !push_constants.is_empty() {
            unsafe {
                device.cmd_push_constants(
                    cmd,
                    pipeline.layout.handle,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }
        }

        unsafe {
            device.cmd_dispatch(cmd, grid_size.0, grid_size.1, grid_size.2);
        }

        unsafe {
            device.end_command_buffer(cmd)
                .context("[ComputeExecutor] Failed to finalize compute commands")?;
        }

        let signal_semaphores = [self.engine.timeline_semaphore.handle];
        let signal_values = [next_value];

        let mut timeline_submit_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(&signal_values);

        let command_buffers = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .push_next(&mut timeline_submit_info)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            device.queue_submit(self.engine.queue.handle, &[submit_info], vk::Fence::null())
                .context("[ComputeExecutor] Failed to submit compute commands to queue")?;
        }

        slot.last_submitted_timeline_value = next_value;
        command_ring.next_slot_idx += 1;

        Ok(ComputeExecutionTask {
            timeline_semaphore: self.engine.timeline_semaphore.handle,
            target_value: next_value,
            device: device.clone(),
        })
    }
}