use enki::{
    enki_compute, enki_vertex, enki_fragment, EnkiStruct,
    init_global_engine,
    GpuDeviceBuffer, GpuTransferManager, BufferUsage,
    GpuWindow, GpuRenderer, GpuGraphicsPipeline,
    ComputeEngineConfig,
};

use ash::vk;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[derive(Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, EnkiStruct)]
#[repr(C)]
pub struct Particle {
    pub position: [f32; 2],
    pub color: [f32; 4],
    pub velocity: [f32; 2],
}

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, EnkiStruct)]
#[repr(C)]
pub struct VertexOutput {
    #[enki(position)]
    pub position: [f32; 2],

    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub speed: f32,

    #[enki(point_size)]
    pub point_size: f32,
}

#[enki_compute]
fn update_particles(my_particle: Particle, dt: f32) {
    let px = my_particle.position[0];
    let py = my_particle.position[1];
    let vx = my_particle.velocity[0];
    let vy = my_particle.velocity[1];

    let dist_sq = (px * px) + (py * py);
    let dist = dist_sq.sqrt();
    let gravity = 0.09 / (dist_sq + 0.04);

    let fx = (0.0 - px) / dist;
    let fy = (0.0 - py) / dist;

    let nvx = (vx + (fx * gravity * dt)) * 0.99;
    let nvy = (vy + (fy * gravity * dt)) * 0.99;

    my_particle.velocity[0] = nvx;
    my_particle.velocity[1] = nvy;
    my_particle.position[0] = px + (nvx * dt);
    my_particle.position[1] = py + (nvy * dt);
}
#[enki_vertex]
fn draw_particles_vs(vert: Particle) -> VertexOutput {
    let px = vert.position[0];
    let py = vert.position[1];
    let vx = vert.velocity[0];
    let vy = vert.velocity[1];

    let speed = ((vx * vx) + (vy * vy)).sqrt();

    let size = 1.0;

    VertexOutput {
        position: [px, py],
        color: vert.color,
        speed: speed,
        point_size: size,
    }
}

#[enki_fragment]
fn draw_particles_fs(varying: VertexOutput) -> [f32; 4] {
    let factor = (varying.speed * 4.0).min(1.0);

    let r = (0.15 * (1.0 - factor)) + (0.0 * factor);
    let g = (0.05 * (1.0 - factor)) + (0.95 * factor);
    let b = (0.65 * (1.0 - factor)) + (0.95 * factor);

    [1.0, 1.0, 1.0, 1.0]
}

fn main() -> anyhow::Result<()> {
    const SIZE: usize = 1_000_000;

    let mut config = ComputeEngineConfig::default();
    config.request_graphics_queue = true;

    config.required_instance_extensions.push(
        std::ffi::CString::from(ash::khr::surface::NAME)
    );

    config.required_instance_extensions.push(
        std::ffi::CString::from(ash::khr::win32_surface::NAME)
    );

    config.required_device_extensions.push(
        std::ffi::CString::from(ash::khr::swapchain::NAME)
    );

    let engine = init_global_engine(config)
        .map_err(|e| anyhow::anyhow!(e))?;

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new()
        .with_title("Enki GPGPU Particles Simulation (Stable Sandbox)")
        .with_inner_size(winit::dpi::PhysicalSize::new(1080, 720))
        .build(&event_loop)
        .unwrap());

    let physical_device = engine.allocator.physical_device;
    let mut gpu_window = GpuWindow::new(
        &engine.instance,
        physical_device,
        &engine.device,
        window.clone(),
        2,
    )?;

    let renderer = GpuRenderer::new(
        &engine.device.logical_device,
        engine.queue.handle,
        engine.queue.family_index,
        4,
    )?;

    let binding_description = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<Particle>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);

    let attribute_descriptions = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(bytemuck::offset_of!(Particle, position) as u32),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(bytemuck::offset_of!(Particle, color) as u32),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(bytemuck::offset_of!(Particle, velocity) as u32),
    ];

    let graphics_pipeline = GpuGraphicsPipeline::new(
        &engine.device.logical_device,
        &[],
        draw_particles_vs(),
        draw_particles_fs(),
        gpu_window.format.format,
        vk::PrimitiveTopology::POINT_LIST,
        0,
        std::slice::from_ref(&binding_description),
        &attribute_descriptions,
    )?;

    let mut initial_particles = Vec::with_capacity(SIZE);
    for i in 0..SIZE {
        let t = i as f32 / SIZE as f32 * 2.0 * std::f32::consts::PI;
        let radius = 0.5;
        initial_particles.push(Particle {
            position: [t.cos() * radius, t.sin() * radius],
            color: [t.sin().abs(), t.cos().abs(), 1.0, 1.0],
            velocity: [-t.sin() * 0.35, t.cos() * 0.35],
        });
    }

    let particle_buffer = GpuDeviceBuffer::<Particle>::new(
        engine.allocator.clone(),
        SIZE,
        BufferUsage::STORAGE_BUFFER | BufferUsage::VERTEX_BUFFER,
    )?;

    let transfer_manager = GpuTransferManager::new(
        engine.allocator.clone(),
        engine.queue.handle,
        engine.command_ring.lock().unwrap().command_pool.handle,
        1024 * 1024 * 4,
    )?;

    transfer_manager.write_buffer(&particle_buffer, 0, &initial_particles)?;

    engine.bind_storage_device_buffer(0, &particle_buffer)?;

    let mut last_time = Instant::now();
    let mut resize_needed = false;

    println!("[Enki Sandbox] Compilation Succeeded. Stable Graphics & Compute Pipeline Active!");

    let start_time = Instant::now();
    let mut last_time = Instant::now();
    let mut resize_needed = false;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    elwt.exit();
                }
                WindowEvent::Resized(_) => {
                    resize_needed = true;
                }
                _ => {}
            }
            Event::AboutToWait => {
                if resize_needed {
                    gpu_window.recreate(physical_device).unwrap();
                    resize_needed = false;
                }

                let current_time = Instant::now();
                let dt = (current_time - last_time).as_secs_f32();
                last_time = current_time;

                let time = start_time.elapsed().as_secs_f32();

                update_particles(&particle_buffer, dt).unwrap();

                let render_res = renderer.draw_frame(&mut gpu_window, |cmd, _target_view, _extent| {
                    unsafe {
                        engine.device.logical_device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            graphics_pipeline.pipeline,
                        );

                        let buffers = [particle_buffer.buffer()];
                        let offsets = [0];
                        engine.device.logical_device.cmd_bind_vertex_buffers(cmd, 0, &buffers, &offsets);

                        engine.device.logical_device.cmd_draw(cmd, SIZE as u32, 1, 0, 0);
                    }
                    Ok(())
                });

                match render_res {
                    Ok(_) => {}
                    Err(e) => {
                        if e.to_string().contains("suboptimal") || e.to_string().contains("out of date") {
                            resize_needed = true;
                        } else {
                            panic!("[Enki Sandbox Error] {}", e);
                        }
                    }
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}