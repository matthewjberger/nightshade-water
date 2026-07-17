//! GPU-driven water: a custom render-graph pass that simulates the height field
//! on ping-pong storage textures with compute shaders and draws the displaced
//! grid, sampling the simulated height and normals directly on the GPU.

use nightshade::prelude::nalgebra_glm;
use nightshade::prelude::wgpu;
use nightshade::render::wgpu::render_configs::RenderInputs;
use nightshade::render::wgpu::rendergraph::{
    PassExecutionContext, PassNode, Result, SubGraphRunCommand,
};
use nightshade::render::wgpu::shader_compose::compile_wgsl;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

const SIM_SIZE: u32 = 256;
const GRID_RESOLUTION: u32 = 200;
const EXTENT: f32 = 6.0;
const WATER_LEVEL: f32 = 0.0;
const HEIGHT_SCALE: f32 = 8.0;
const FLOOR_Y: f32 = -5.8;
/// Fixed simulation rate so ripple speed is independent of the frame rate.
const STEP_HZ: f32 = 60.0;
const MAX_STEPS_PER_FRAME: u32 = 4;

/// Interaction state written by the app each frame and read by the pass.
#[derive(Clone, Copy)]
pub struct WaterParams {
    pub drop_center: [f32; 2],
    pub drop_radius: f32,
    pub drop_strength: f32,
    pub drop_active: bool,
    pub sphere_old: [f32; 3],
    pub sphere_new: [f32; 3],
    pub sphere_radius: f32,
    /// Requests the pass to clear the height field back to a flat pool.
    pub reset: bool,
}

impl Default for WaterParams {
    fn default() -> Self {
        Self {
            drop_center: [0.5, 0.5],
            drop_radius: 0.03,
            drop_strength: 0.0,
            drop_active: false,
            sphere_old: [0.0, -1.0, 0.0],
            sphere_new: [0.0, -1.0, 0.0],
            sphere_radius: 0.25,
            reset: false,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SimUniform {
    drop: [f32; 4],
    flags: [f32; 4],
    sphere_old: [f32; 4],
    sphere_new: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    sun_dir: [f32; 4],
    sun_color: [f32; 4],
    shallow: [f32; 4],
    deep: [f32; 4],
    params: [f32; 4],
}

pub struct WaterGpuPass {
    params: Arc<Mutex<WaterParams>>,
    sim_uniform: wgpu::Buffer,
    render_uniform: wgpu::Buffer,
    clear_pipeline: wgpu::ComputePipeline,
    update_pipeline: wgpu::ComputePipeline,
    normals_pipeline: wgpu::ComputePipeline,
    update_bind_group: wgpu::BindGroup,
    normals_bind_group: wgpu::BindGroup,
    cleared: bool,
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    caustics_pipeline: wgpu::RenderPipeline,
    caustics_vertex_buffer: wgpu::Buffer,
    caustics_index_buffer: wgpu::Buffer,
    step_accumulator: f32,
}

fn build_grid(resolution: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let segments = resolution.max(2);
    let count = segments + 1;
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    for z in 0..count {
        for x in 0..count {
            let fx = x as f32 / segments as f32 * 2.0 - 1.0;
            let fz = z as f32 / segments as f32 * 2.0 - 1.0;
            vertices.push([fx, fz, 0.0]);
        }
    }
    let mut indices = Vec::new();
    for z in 0..segments {
        for x in 0..segments {
            let top_left = z * count + x;
            let top_right = top_left + 1;
            let bottom_left = top_left + count;
            let bottom_right = bottom_left + 1;
            indices.extend_from_slice(&[
                top_left,
                bottom_left,
                top_right,
                top_right,
                bottom_left,
                bottom_right,
            ]);
        }
    }

    let mut skirt = |ax: f32, az: f32, bx: f32, bz: f32| {
        let base = vertices.len() as u32;
        vertices.push([ax, az, 0.0]);
        vertices.push([bx, bz, 0.0]);
        vertices.push([bx, bz, 1.0]);
        vertices.push([ax, az, 1.0]);
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };
    for i in 0..segments {
        let t0 = i as f32 / segments as f32 * 2.0 - 1.0;
        let t1 = (i + 1) as f32 / segments as f32 * 2.0 - 1.0;
        skirt(t0, 1.0, t1, 1.0);
        skirt(t1, -1.0, t0, -1.0);
        skirt(1.0, t1, 1.0, t0);
        skirt(-1.0, t0, -1.0, t1);
    }

    (vertices, indices)
}

fn make_state_texture(device: &wgpu::Device, label: &str) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: SIM_SIZE,
            height: SIM_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

impl WaterGpuPass {
    pub fn new(device: &wgpu::Device, params: Arc<Mutex<WaterParams>>) -> Self {
        let state_a = make_state_texture(device, "Water State A");
        let state_b = make_state_texture(device, "Water State B");

        let sim_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Sim Uniform"),
            size: std::mem::size_of::<SimUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let render_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Render Uniform"),
            size: std::mem::size_of::<RenderUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Compute Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sim_shader = compile_wgsl(
            device,
            "water_sim.wgsl",
            include_str!("shaders/water_sim.wgsl"),
        );
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Water Compute Pipeline Layout"),
                bind_group_layouts: &[Some(&compute_layout)],
                immediate_size: 0,
            });
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Water Clear Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &sim_shader,
            entry_point: Some("clear"),
            compilation_options: Default::default(),
            cache: None,
        });
        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Water Update Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &sim_shader,
            entry_point: Some("update"),
            compilation_options: Default::default(),
            cache: None,
        });
        let normals_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Water Normals Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &sim_shader,
            entry_point: Some("normals"),
            compilation_options: Default::default(),
            cache: None,
        });

        let update_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water Update Bind Group"),
            layout: &compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&state_a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&state_b),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sim_uniform.as_entire_binding(),
                },
            ],
        });
        let normals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water Normals Bind Group"),
            layout: &compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&state_b),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&state_a),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sim_uniform.as_entire_binding(),
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Water State Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Render Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let render_shader = compile_wgsl(
            device,
            "water_surface.wgsl",
            include_str!("shaders/water_surface.wgsl"),
        );
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Water Render Pipeline Layout"),
                bind_group_layouts: &[Some(&render_layout)],
                immediate_size: 0,
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water Render Bind Group"),
            layout: &render_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: render_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&state_a),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let (vertices, indices) = build_grid(GRID_RESOLUTION);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Grid Vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Grid Indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let caustics_shader = compile_wgsl(
            device,
            "caustics.wgsl",
            include_str!("shaders/caustics.wgsl"),
        );
        let caustics_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Caustics Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &caustics_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &caustics_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let caustics_quad: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let caustics_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
        let caustics_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Caustics Vertices"),
            contents: bytemuck::cast_slice(&caustics_quad),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let caustics_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Caustics Indices"),
            contents: bytemuck::cast_slice(&caustics_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            params,
            sim_uniform,
            render_uniform,
            clear_pipeline,
            update_pipeline,
            normals_pipeline,
            update_bind_group,
            normals_bind_group,
            cleared: false,
            render_pipeline,
            render_bind_group,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            caustics_pipeline,
            caustics_vertex_buffer,
            caustics_index_buffer,
            step_accumulator: 0.0,
        }
    }
}

impl PassNode<RenderInputs> for WaterGpuPass {
    fn name(&self) -> &str {
        "water_gpu_pass"
    }

    fn reads(&self) -> Vec<&str> {
        vec![]
    }

    fn writes(&self) -> Vec<&str> {
        vec![]
    }

    fn reads_writes(&self) -> Vec<&str> {
        vec!["color", "depth"]
    }

    fn execute<'r, 'e>(
        &mut self,
        context: PassExecutionContext<'r, 'e, RenderInputs>,
    ) -> Result<Vec<SubGraphRunCommand<'r>>> {
        let configs = context.configs;
        let camera = match configs.scene.render_view.as_ref() {
            Some(matrices) => matrices,
            None => return Ok(context.into_sub_graph_commands()),
        };

        let params = {
            let mut guard = self.params.lock().unwrap();
            let snapshot = *guard;
            guard.reset = false;
            snapshot
        };

        let drop = params;

        let sim = SimUniform {
            drop: [
                drop.drop_center[0],
                drop.drop_center[1],
                drop.drop_radius,
                drop.drop_strength,
            ],
            flags: [
                if drop.drop_active { 1.0 } else { 0.0 },
                1.0 / SIM_SIZE as f32,
                0.0,
                0.0,
            ],
            sphere_old: [
                drop.sphere_old[0],
                drop.sphere_old[1],
                drop.sphere_old[2],
                drop.sphere_radius,
            ],
            sphere_new: [
                drop.sphere_new[0],
                drop.sphere_new[1],
                drop.sphere_new[2],
                drop.sphere_radius,
            ],
        };
        context
            .queue
            .write_buffer(&self.sim_uniform, 0, bytemuck::bytes_of(&sim));

        let view_proj: [[f32; 4]; 4] = (camera.projection * camera.view).into();
        let camera_position = camera.camera_position;
        let render = RenderUniform {
            view_proj,
            camera_pos: [camera_position.x, camera_position.y, camera_position.z, 1.0],
            sun_dir: {
                let sun = nalgebra_glm::normalize(&nalgebra_glm::vec3(0.3, 1.0, 0.2));
                [sun.x, sun.y, sun.z, 0.0]
            },
            sun_color: [1.0, 0.96, 0.88, 0.0],
            shallow: [0.16, 0.46, 0.52, 0.0],
            deep: [0.03, 0.16, 0.24, 0.0],
            params: [EXTENT, WATER_LEVEL, HEIGHT_SCALE, FLOOR_Y],
        };
        context
            .queue
            .write_buffer(&self.render_uniform, 0, bytemuck::bytes_of(&render));

        let workgroups = SIM_SIZE.div_ceil(8);

        if !self.cleared || params.reset {
            self.cleared = true;
            self.step_accumulator = 0.0;
            for bind_group in [&self.update_bind_group, &self.normals_bind_group] {
                let mut compute =
                    context
                        .encoder
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Water Clear"),
                            timestamp_writes: None,
                        });
                compute.set_pipeline(&self.clear_pipeline);
                compute.set_bind_group(0, bind_group, &[]);
                compute.dispatch_workgroups(workgroups, workgroups, 1);
            }
        }

        self.step_accumulator += configs.view.delta_time.max(0.0);
        let step_dt = 1.0 / STEP_HZ;
        let steps = ((self.step_accumulator / step_dt).floor() as u32).min(MAX_STEPS_PER_FRAME);
        self.step_accumulator = (self.step_accumulator - steps as f32 * step_dt).max(0.0);

        for _ in 0..steps {
            {
                let mut compute =
                    context
                        .encoder
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Water Update"),
                            timestamp_writes: None,
                        });
                compute.set_pipeline(&self.update_pipeline);
                compute.set_bind_group(0, &self.update_bind_group, &[]);
                compute.dispatch_workgroups(workgroups, workgroups, 1);
            }
            {
                let mut compute =
                    context
                        .encoder
                        .begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Water Normals"),
                            timestamp_writes: None,
                        });
                compute.set_pipeline(&self.normals_pipeline);
                compute.set_bind_group(0, &self.normals_bind_group, &[]);
                compute.dispatch_workgroups(workgroups, workgroups, 1);
            }
        }

        let (color_view, color_load, color_store) = context.get_color_attachment("color")?;
        let (depth_view, depth_load, depth_store) = context.get_depth_attachment("depth")?;
        {
            let mut render_pass = context
                .encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Water Render"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: color_load,
                            store: color_store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: depth_load,
                            store: depth_store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            render_pass.set_bind_group(0, &self.render_bind_group, &[]);

            render_pass.set_pipeline(&self.caustics_pipeline);
            render_pass.set_vertex_buffer(0, self.caustics_vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                self.caustics_index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            render_pass.draw_indexed(0..6, 0, 0..1);

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.index_count, 0, 0..1);
        }

        Ok(context.into_sub_graph_commands())
    }
}
