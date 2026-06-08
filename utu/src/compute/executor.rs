use ash::vk;
use anyhow::{Result, Context};
use std::time::Duration;

use crate::vulkan::VulkanFence;
use crate::compute::engine::ComputeEngine;
use crate::compute::pipeline::ComputePipeline;

pub struct ComputeExecutionTask {
    pub fence: VulkanFence,
    cmd: vk::CommandBuffer,
    pool_handle: vk::CommandPool,
    device: ash::Device,
}

impl ComputeExecutionTask {
    pub fn wait(&self, timeout: Duration) -> Result<()> {
        self.fence.wait(timeout)
    }
}

impl Drop for ComputeExecutionTask {
    fn drop(&mut self) {
        unsafe {
            let _ = self.fence.wait(Duration::from_secs(3600));
            self.device.free_command_buffers(self.pool_handle, &[self.cmd]);
        }
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

        let cmd = self.engine.command_pool.allocate_buffer(vk::CommandBufferLevel::PRIMARY)?;

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
                .context("[ComputeExecutor] Failed to finalize compute commands recording")?;
        }

        let fence = VulkanFence::new(device, false)
            .context("[ComputeExecutor] Failed to create sync fence")?;

        let command_buffers = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&command_buffers);

        unsafe {
            device.queue_submit(self.engine.queue.handle, &[submit_info], fence.handle)
                .context("[ComputeExecutor] Failed to submit compute commands to queue")?;
        }

        Ok(ComputeExecutionTask {
            fence,
            cmd,
            pool_handle: self.engine.command_pool.handle,
            device: device.clone(),
        })
    }
}