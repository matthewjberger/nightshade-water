//! GPU-driven water rendered the way Evan Wallace's WebGL Water does it: the
//! height field is simulated on ping-pong storage textures with compute
//! shaders, a caustics texture is projected from the surface each frame, and the
//! pool, water, and ball are drawn by ray tracing in the normalized pool space
//! (interior x and z in [-1, 1], floor at y = -1, water rest at y = 0).

use nightshade::prelude::image;
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
const CAUSTIC_SIZE: u32 = 1024;
const GRID_RESOLUTION: u32 = 200;
const SPHERE_STACKS: u32 = 24;
const SPHERE_SLICES: u32 = 48;
const RIM_Y: f32 = 2.0 / 12.0;
const FLOOR_Y: f32 = -1.0;
const STEP_HZ: f32 = 60.0;
const MAX_STEPS_PER_FRAME: u32 = 4;

const HELPERS: &str = include_str!("shaders/helpers.wgsl");
const TILE_BYTES: &[u8] = include_bytes!("../assets/tiles.jpg");
const SKY_POS_X: &[u8] = include_bytes!("../assets/xpos.jpg");
const SKY_NEG_X: &[u8] = include_bytes!("../assets/xneg.jpg");
const SKY_POS_Y: &[u8] = include_bytes!("../assets/ypos.jpg");
const SKY_POS_Z: &[u8] = include_bytes!("../assets/zpos.jpg");
const SKY_NEG_Z: &[u8] = include_bytes!("../assets/zneg.jpg");

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
    pub sphere_visible: bool,
    pub reset: bool,
}

impl Default for WaterParams {
    fn default() -> Self {
        Self {
            drop_center: [0.5, 0.5],
            drop_radius: 0.03,
            drop_strength: 0.0,
            drop_active: false,
            sphere_old: [-0.4, -0.75, 0.2],
            sphere_new: [-0.4, -0.75, 0.2],
            sphere_radius: 0.25,
            sphere_visible: true,
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
    eye: [f32; 4],
    light: [f32; 4],
    sphere: [f32; 4],
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
    step_accumulator: f32,
    caustic_view: wgpu::TextureView,
    caustics_pipeline: wgpu::RenderPipeline,
    caustics_bind_group: wgpu::BindGroup,
    pool_pipeline: wgpu::RenderPipeline,
    water_above_pipeline: wgpu::RenderPipeline,
    water_under_pipeline: wgpu::RenderPipeline,
    sphere_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    plane_vertex_buffer: wgpu::Buffer,
    plane_index_buffer: wgpu::Buffer,
    plane_index_count: u32,
    box_vertex_buffer: wgpu::Buffer,
    box_index_buffer: wgpu::Buffer,
    box_index_count: u32,
    sphere_vertex_buffer: wgpu::Buffer,
    sphere_index_buffer: wgpu::Buffer,
    sphere_index_count: u32,
    pending: Option<PendingUploads>,
}

fn build_plane(resolution: u32) -> (Vec<[f32; 2]>, Vec<u32>) {
    let segments = resolution.max(2);
    let count = segments + 1;
    let mut vertices: Vec<[f32; 2]> = Vec::new();
    for z in 0..count {
        for x in 0..count {
            let fx = x as f32 / segments as f32 * 2.0 - 1.0;
            let fz = z as f32 / segments as f32 * 2.0 - 1.0;
            vertices.push([fx, fz]);
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
    (vertices, indices)
}

fn build_pool_box() -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut quad = |corners: [[f32; 3]; 4]| {
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&corners);
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };
    quad([
        [-1.0, FLOOR_Y, -1.0],
        [-1.0, FLOOR_Y, 1.0],
        [1.0, FLOOR_Y, 1.0],
        [1.0, FLOOR_Y, -1.0],
    ]);
    quad([
        [1.0, FLOOR_Y, -1.0],
        [1.0, FLOOR_Y, 1.0],
        [1.0, RIM_Y, 1.0],
        [1.0, RIM_Y, -1.0],
    ]);
    quad([
        [-1.0, RIM_Y, -1.0],
        [-1.0, RIM_Y, 1.0],
        [-1.0, FLOOR_Y, 1.0],
        [-1.0, FLOOR_Y, -1.0],
    ]);
    quad([
        [-1.0, FLOOR_Y, 1.0],
        [-1.0, RIM_Y, 1.0],
        [1.0, RIM_Y, 1.0],
        [1.0, FLOOR_Y, 1.0],
    ]);
    quad([
        [1.0, FLOOR_Y, -1.0],
        [1.0, RIM_Y, -1.0],
        [-1.0, RIM_Y, -1.0],
        [-1.0, FLOOR_Y, -1.0],
    ]);
    (vertices, indices)
}

fn build_sphere() -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    for stack in 0..=SPHERE_STACKS {
        let phi = std::f32::consts::PI * stack as f32 / SPHERE_STACKS as f32;
        let y = phi.cos();
        let ring = phi.sin();
        for slice in 0..=SPHERE_SLICES {
            let theta = std::f32::consts::TAU * slice as f32 / SPHERE_SLICES as f32;
            vertices.push([ring * theta.cos(), y, ring * theta.sin()]);
        }
    }
    let mut indices = Vec::new();
    let columns = SPHERE_SLICES + 1;
    for stack in 0..SPHERE_STACKS {
        for slice in 0..SPHERE_SLICES {
            let first = stack * columns + slice;
            let second = first + columns;
            indices.extend_from_slice(&[first, second, first + 1, first + 1, second, second + 1]);
        }
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

struct DecodedImage {
    bytes: Vec<u8>,
    width: u32,
    height: u32,
}

fn decode_image(bytes: &[u8]) -> DecodedImage {
    let decoded = image::load_from_memory(bytes)
        .expect("decode texture")
        .to_rgba8();
    let (width, height) = decoded.dimensions();
    DecodedImage {
        bytes: decoded.into_raw(),
        width,
        height,
    }
}

/// Textures whose pixels are uploaded on the first frame, when a queue is
/// available (the render-graph config hook only hands us a device).
struct PendingUploads {
    tile_texture: wgpu::Texture,
    tile: DecodedImage,
    sky_texture: wgpu::Texture,
    sky_faces: Vec<DecodedImage>,
}

fn write_layer(queue: &wgpu::Queue, texture: &wgpu::Texture, layer: u32, image: &DecodedImage) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        &image.bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * image.width),
            rows_per_image: Some(image.height),
        },
        wgpu::Extent3d {
            width: image.width,
            height: image.height,
            depth_or_array_layers: 1,
        },
    );
}

fn create_2d_texture(device: &wgpu::Device, label: &str, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn create_sky_cube(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Water Sky Cube"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

struct PipelineConfig<'a> {
    shader: &'a wgpu::ShaderModule,
    fragment_entry: &'a str,
    vertex_stride: u64,
    vertex_format: wgpu::VertexFormat,
    cull_mode: Option<wgpu::Face>,
    label: &'a str,
}

fn render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    config: PipelineConfig,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(config.label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: config.shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: config.vertex_stride,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: config.vertex_format,
                    offset: 0,
                    shader_location: 0,
                }],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: config.shader,
            entry_point: Some(config.fragment_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: config.cull_mode,
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
    })
}

impl WaterGpuPass {
    pub fn new(device: &wgpu::Device, params: Arc<Mutex<WaterParams>>) -> Self {
        let state_a = make_state_texture(device, "Water State A");
        let state_b = make_state_texture(device, "Water State B");

        let tile = decode_image(TILE_BYTES);
        let tile_texture = create_2d_texture(device, "Water Tiles", tile.width, tile.height);
        let tile_view = tile_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sky_faces: Vec<DecodedImage> =
            [SKY_POS_X, SKY_NEG_X, SKY_POS_Y, SKY_POS_Y, SKY_POS_Z, SKY_NEG_Z]
                .iter()
                .map(|bytes| decode_image(bytes))
                .collect();
        let sky_texture = create_sky_cube(device, sky_faces[0].width, sky_faces[0].height);
        let sky_view = sky_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Water Sky Cube View"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let pending = Some(PendingUploads {
            tile_texture,
            tile,
            sky_texture,
            sky_faces,
        });

        let caustic_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water Caustic Texture"),
            size: wgpu::Extent3d {
                width: CAUSTIC_SIZE,
                height: CAUSTIC_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let caustic_view = caustic_texture.create_view(&wgpu::TextureViewDescriptor::default());

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

        let (clear_pipeline, update_pipeline, normals_pipeline, update_bind_group, normals_bind_group) =
            build_compute(device, &state_a, &state_b, &sim_uniform);

        let clamp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Water Clamp Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let repeat_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Water Repeat Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let cube_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Water Cube Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let render_layout = build_render_layout(device);
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Render Pipeline Layout"),
            bind_group_layouts: &[Some(&render_layout)],
            immediate_size: 0,
        });

        let make_render_bind_group = |water: &wgpu::TextureView, caustic: &wgpu::TextureView, label: &str| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &render_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: render_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(water),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&tile_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(caustic),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&sky_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&clamp_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&repeat_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(&cube_sampler),
                    },
                ],
            })
        };
        let render_bind_group = make_render_bind_group(&state_a, &caustic_view, "Water Render Bind Group");
        let caustics_bind_group = make_render_bind_group(&state_a, &state_b, "Water Caustics Bind Group");

        let helper_source = |main: &str| format!("{HELPERS}\n{main}");
        let pool_shader = compile_wgsl(device, "pool.wgsl", &helper_source(include_str!("shaders/pool.wgsl")));
        let sphere_shader = compile_wgsl(device, "sphere.wgsl", &helper_source(include_str!("shaders/sphere.wgsl")));
        let surface_shader = compile_wgsl(device, "water_surface.wgsl", &helper_source(include_str!("shaders/water_surface.wgsl")));
        let caustics_shader = compile_wgsl(device, "caustics.wgsl", &helper_source(include_str!("shaders/caustics.wgsl")));

        let pool_pipeline = render_pipeline(
            device,
            &render_pipeline_layout,
            PipelineConfig {
                shader: &pool_shader,
                fragment_entry: "fs_main",
                vertex_stride: 12,
                vertex_format: wgpu::VertexFormat::Float32x3,
                cull_mode: Some(wgpu::Face::Back),
                label: "Water Pool Pipeline",
            },
        );
        let sphere_pipeline = render_pipeline(
            device,
            &render_pipeline_layout,
            PipelineConfig {
                shader: &sphere_shader,
                fragment_entry: "fs_main",
                vertex_stride: 12,
                vertex_format: wgpu::VertexFormat::Float32x3,
                cull_mode: None,
                label: "Water Sphere Pipeline",
            },
        );
        let water_above_pipeline = render_pipeline(
            device,
            &render_pipeline_layout,
            PipelineConfig {
                shader: &surface_shader,
                fragment_entry: "fs_above",
                vertex_stride: 8,
                vertex_format: wgpu::VertexFormat::Float32x2,
                cull_mode: Some(wgpu::Face::Back),
                label: "Water Above Pipeline",
            },
        );
        let water_under_pipeline = render_pipeline(
            device,
            &render_pipeline_layout,
            PipelineConfig {
                shader: &surface_shader,
                fragment_entry: "fs_under",
                vertex_stride: 8,
                vertex_format: wgpu::VertexFormat::Float32x2,
                cull_mode: Some(wgpu::Face::Front),
                label: "Water Under Pipeline",
            },
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
                    blend: None,
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let (plane_vertices, plane_indices) = build_plane(GRID_RESOLUTION);
        let plane_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Plane Vertices"),
            contents: bytemuck::cast_slice(&plane_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let plane_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Plane Indices"),
            contents: bytemuck::cast_slice(&plane_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let (box_vertices, box_indices) = build_pool_box();
        let box_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Pool Vertices"),
            contents: bytemuck::cast_slice(&box_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let box_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Pool Indices"),
            contents: bytemuck::cast_slice(&box_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let (sphere_vertices, sphere_indices) = build_sphere();
        let sphere_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Sphere Vertices"),
            contents: bytemuck::cast_slice(&sphere_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sphere_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Sphere Indices"),
            contents: bytemuck::cast_slice(&sphere_indices),
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
            step_accumulator: 0.0,
            caustic_view,
            caustics_pipeline,
            caustics_bind_group,
            pool_pipeline,
            water_above_pipeline,
            water_under_pipeline,
            sphere_pipeline,
            render_bind_group,
            plane_vertex_buffer,
            plane_index_buffer,
            plane_index_count: plane_indices.len() as u32,
            box_vertex_buffer,
            box_index_buffer,
            box_index_count: box_indices.len() as u32,
            sphere_vertex_buffer,
            sphere_index_buffer,
            sphere_index_count: sphere_indices.len() as u32,
            pending,
        }
    }
}

fn build_compute(
    device: &wgpu::Device,
    state_a: &wgpu::TextureView,
    state_b: &wgpu::TextureView,
    sim_uniform: &wgpu::Buffer,
) -> (
    wgpu::ComputePipeline,
    wgpu::ComputePipeline,
    wgpu::ComputePipeline,
    wgpu::BindGroup,
    wgpu::BindGroup,
) {
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

    let sim_shader = compile_wgsl(device, "water_sim.wgsl", include_str!("shaders/water_sim.wgsl"));
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Water Compute Pipeline Layout"),
        bind_group_layouts: &[Some(&compute_layout)],
        immediate_size: 0,
    });
    let make_compute = |entry: &str, label: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&compute_pipeline_layout),
            module: &sim_shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let clear_pipeline = make_compute("clear", "Water Clear Pipeline");
    let update_pipeline = make_compute("update", "Water Update Pipeline");
    let normals_pipeline = make_compute("normals", "Water Normals Pipeline");

    let make_bind_group = |src: &wgpu::TextureView, dst: &wgpu::TextureView, label: &str| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(dst),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sim_uniform.as_entire_binding(),
                },
            ],
        })
    };
    let update_bind_group = make_bind_group(state_a, state_b, "Water Update Bind Group");
    let normals_bind_group = make_bind_group(state_b, state_a, "Water Normals Bind Group");

    (
        clear_pipeline,
        update_pipeline,
        normals_pipeline,
        update_bind_group,
        normals_bind_group,
    )
}

fn build_render_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture_entry = |binding: u32, visibility: wgpu::ShaderStages, dimension: wgpu::TextureViewDimension| {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: dimension,
                multisampled: false,
            },
            count: None,
        }
    };
    let sampler_entry = |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            texture_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT, wgpu::TextureViewDimension::D2),
            texture_entry(2, wgpu::ShaderStages::FRAGMENT, wgpu::TextureViewDimension::D2),
            texture_entry(3, wgpu::ShaderStages::FRAGMENT, wgpu::TextureViewDimension::D2),
            texture_entry(4, wgpu::ShaderStages::FRAGMENT, wgpu::TextureViewDimension::Cube),
            sampler_entry(5, wgpu::ShaderStages::VERTEX_FRAGMENT),
            sampler_entry(6, wgpu::ShaderStages::FRAGMENT),
            sampler_entry(7, wgpu::ShaderStages::FRAGMENT),
        ],
    })
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

        if let Some(pending) = self.pending.take() {
            write_layer(context.queue, &pending.tile_texture, 0, &pending.tile);
            for (layer, face) in pending.sky_faces.iter().enumerate() {
                write_layer(context.queue, &pending.sky_texture, layer as u32, face);
            }
        }

        let params = {
            let mut guard = self.params.lock().unwrap();
            let snapshot = *guard;
            guard.reset = false;
            snapshot
        };

        let sim = SimUniform {
            drop: [
                params.drop_center[0],
                params.drop_center[1],
                params.drop_radius,
                params.drop_strength,
            ],
            flags: [
                if params.drop_active { 1.0 } else { 0.0 },
                1.0 / SIM_SIZE as f32,
                0.0,
                0.0,
            ],
            sphere_old: [
                params.sphere_old[0],
                params.sphere_old[1],
                params.sphere_old[2],
                params.sphere_radius,
            ],
            sphere_new: [
                params.sphere_new[0],
                params.sphere_new[1],
                params.sphere_new[2],
                params.sphere_radius,
            ],
        };
        context
            .queue
            .write_buffer(&self.sim_uniform, 0, bytemuck::bytes_of(&sim));

        let view_proj: [[f32; 4]; 4] = (camera.projection * camera.view).into();
        let eye = camera.camera_position;
        let light = nalgebra_glm::normalize(&nalgebra_glm::vec3(2.0, 2.0, -1.0));
        let render = RenderUniform {
            view_proj,
            eye: [eye.x, eye.y, eye.z, 1.0],
            light: [light.x, light.y, light.z, 0.0],
            sphere: [
                params.sphere_new[0],
                params.sphere_new[1],
                params.sphere_new[2],
                params.sphere_radius,
            ],
        };
        context
            .queue
            .write_buffer(&self.render_uniform, 0, bytemuck::bytes_of(&render));

        let workgroups = SIM_SIZE.div_ceil(8);

        if !self.cleared || params.reset {
            self.cleared = true;
            self.step_accumulator = 0.0;
            for bind_group in [&self.update_bind_group, &self.normals_bind_group] {
                let mut compute = context
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
                let mut compute = context
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
                let mut compute = context
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

        {
            let mut caustics = context
                .encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Water Caustics"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.caustic_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            caustics.set_pipeline(&self.caustics_pipeline);
            caustics.set_bind_group(0, &self.caustics_bind_group, &[]);
            caustics.set_vertex_buffer(0, self.plane_vertex_buffer.slice(..));
            caustics.set_index_buffer(self.plane_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            caustics.draw_indexed(0..self.plane_index_count, 0, 0..1);
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

            render_pass.set_pipeline(&self.pool_pipeline);
            render_pass.set_vertex_buffer(0, self.box_vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.box_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.box_index_count, 0, 0..1);

            render_pass.set_vertex_buffer(0, self.plane_vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.plane_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.set_pipeline(&self.water_under_pipeline);
            render_pass.draw_indexed(0..self.plane_index_count, 0, 0..1);
            render_pass.set_pipeline(&self.water_above_pipeline);
            render_pass.draw_indexed(0..self.plane_index_count, 0, 0..1);

            if params.sphere_visible {
                render_pass.set_pipeline(&self.sphere_pipeline);
                render_pass.set_vertex_buffer(0, self.sphere_vertex_buffer.slice(..));
                render_pass.set_index_buffer(self.sphere_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.sphere_index_count, 0, 0..1);
            }
        }

        Ok(context.into_sub_graph_commands())
    }
}
