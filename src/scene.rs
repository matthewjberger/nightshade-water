//! Scene construction: the tiled pool, the beach ball, HDR environment, camera,
//! and help. The water itself is drawn by the GPU pass in `water_pass.rs`.

use nightshade::prelude::*;
use nightshade::render::material::{Material, TextureTransform};
use nightshade::render::texture_data::{SamplerSettings, TextureUsage};

/// World half-width of the pool along X and Z.
pub const POOL_HALF: f32 = 6.0;
/// Rest height of the water surface.
pub const WATER_Y: f32 = 0.0;
/// Uniform factor mapping normalized pool space to world space.
pub const WORLD_SCALE: f32 = 6.0;
/// Sphere radius in normalized pool space (matches the original demo).
pub const SPHERE_SIM_RADIUS: f32 = 0.25;
/// Sphere radius in world space.
pub const SPHERE_RADIUS: f32 = SPHERE_SIM_RADIUS * WORLD_SCALE;

const TILE_TEXTURE: &str = "pool_tiles";
const TILE_NORMAL_TEXTURE: &str = "pool_tiles_normal";
const TILE_BYTES: &[u8] = include_bytes!("../assets/tiles.jpg");
const BALL_GLB: &[u8] = include_bytes!("../assets/beach_ball.glb");
const ENV_HDR: &[u8] = include_bytes!("../assets/env.hdr");
/// Visual scale of the ball model. The glTF is authored at unit radius, so this
/// tracks the physics radius and the two stay in sync.
const BALL_SCALE: f32 = SPHERE_RADIUS;

/// Loads the HDR skybox and spawns the key light. Render settings are set by
/// the initialize system through its resource params.
pub fn load_environment(world: &mut World) {
    load_hdr_skybox(world, ENV_HDR.to_vec());

    let sun = spawn_sun(world);
    if let Some(light) = world.get_mut::<nightshade::ecs::light::components::Light>(sun) {
        light.intensity = 2.0;
        light.color = Vec3::new(1.0, 0.97, 0.9);
    }
}

/// Uploads the tile albedo and a normal map derived from it.
pub fn load_textures(world: &mut World) {
    load_texture_pack_from_image_bytes(
        world,
        &[(TILE_TEXTURE, TILE_BYTES)],
        TextureUsage::Color,
        SamplerSettings::DEFAULT,
    );
    let (normal_rgba, width, height) = generate_normal_map(TILE_BYTES);
    queue_decoded_texture(
        world,
        TILE_NORMAL_TEXTURE.to_string(),
        normal_rgba,
        width,
        height,
        TextureUsage::Linear,
        SamplerSettings::DEFAULT,
    );
}

/// Builds the tiled pool box: floor and two walls, open on the near sides.
/// Returns the two wall entities so their visibility can be toggled.
pub fn spawn_pool(world: &mut World) -> Vec<Entity> {
    let floor = spawn_mesh(
        world,
        "Cube",
        Vec3::new(0.0, -POOL_HALF - 0.1, 0.0),
        Vec3::new(POOL_HALF * 2.0 + 1.0, 0.4, POOL_HALF * 2.0 + 1.0),
    );
    set_material(world, floor, "pool_floor".to_string(), tile_material(1.5));

    let wall_height = POOL_HALF + 0.6;
    let wall_center = -POOL_HALF + wall_height * 0.5;
    let span = POOL_HALF * 2.0 + 0.8;
    let walls = [
        (
            "pool_wall_north",
            Vec3::new(0.0, wall_center, POOL_HALF + 0.3),
            Vec3::new(span, wall_height, 0.4),
        ),
        (
            "pool_wall_east",
            Vec3::new(POOL_HALF + 0.3, wall_center, 0.0),
            Vec3::new(0.4, wall_height, span),
        ),
    ];
    walls
        .into_iter()
        .map(|(name, position, scale)| {
            let wall = spawn_mesh(world, "Cube", position, scale);
            set_material(world, wall, name.to_string(), tile_material(1.0));
            wall
        })
        .collect()
}

/// Spawns the beach-ball glTF at a world position and returns its root entity.
pub fn spawn_ball(world: &mut World, position: Vec3) -> Entity {
    if let Ok(mut result) = import_gltf_from_bytes(BALL_GLB) {
        nightshade::ecs::loading::queue_gltf_load(world, &mut result);
        if let Some(prefab) = result.prefabs.first() {
            let entity = nightshade::ecs::prefab::spawn_prefab(world, prefab, position);
            if let Some(transform) = world.get_mut::<LocalTransform>(entity) {
                transform.scale = Vec3::new(BALL_SCALE, BALL_SCALE, BALL_SCALE);
            }
            return entity;
        }
    }
    Entity::default()
}

/// Spawns the orbit camera and returns its entity. The initialize system
/// activates it through its `ActiveCamera` param.
pub fn spawn_camera(world: &mut World) -> Entity {
    nightshade::ecs::camera::commands::spawn_pan_orbit_camera_at(
        world,
        Vec3::new(0.0, -1.0, 0.0),
        22.0,
        0.7 + std::f32::consts::PI,
        0.5,
        "Camera".to_string(),
    )
}

/// Handles to the interactive widgets in the control panel.
pub struct Controls {
    pub rain: Entity,
    pub walls: Entity,
    pub ball: Entity,
    pub reset: Entity,
}

/// Builds the retained-UI control panel and returns its widget handles.
pub fn spawn_controls(world: &mut World) -> Controls {
    let mut tree = UiTreeBuilder::new(world);
    let panel = tree.add_floating_panel(
        "water_controls",
        "Water",
        Rect {
            min: Vec2::new(16.0, 16.0),
            max: Vec2::new(216.0, 216.0),
        },
    );
    let content = widget::<UiPanelData>(tree.world_mut(), panel)
        .map(|data| data.content_entity)
        .unwrap_or(panel);
    let mut controls = Controls {
        rain: Entity::default(),
        walls: Entity::default(),
        ball: Entity::default(),
        reset: Entity::default(),
    };
    tree.in_parent(content, |tree| {
        controls.rain = tree.add_checkbox("Rain", false);
        controls.walls = tree.add_checkbox("Walls", true);
        controls.ball = tree.add_checkbox("Ball", true);
        controls.reset = tree.add_button("Reset");
    });
    tree.finish();
    controls
}

/// Spawns a rain particle emitter above the pool, initially disabled. The drops
/// fall over the water surface where the height field reacts to their impact.
pub fn spawn_rain_emitter(world: &mut World) -> Entity {
    use nightshade::render::particles::{
        ColorGradient, EmitterShape, EmitterType, ParticleEmitter,
    };
    let position = Vec3::new(0.0, WORLD_SCALE * 1.4, 0.0);
    let emitter = ParticleEmitter {
        emitter_type: EmitterType::Sparks,
        shape: EmitterShape::Box {
            half_extents: Vec3::new(POOL_HALF, 0.1, POOL_HALF),
        },
        position,
        direction: Vec3::new(0.0, -1.0, 0.0),
        spawn_rate: 450.0,
        burst_count: 0,
        particle_lifetime_min: 0.7,
        particle_lifetime_max: 0.95,
        initial_velocity_min: 8.0,
        initial_velocity_max: 11.0,
        velocity_spread: 0.03,
        gravity: Vec3::new(0.0, -20.0, 0.0),
        drag: 0.0,
        size_start: 0.16,
        size_end: 0.09,
        color_gradient: ColorGradient {
            colors: vec![
                (0.0, Vec4::new(0.85, 0.93, 1.0, 1.0)),
                (0.7, Vec4::new(0.7, 0.85, 1.0, 0.95)),
                (1.0, Vec4::new(0.6, 0.8, 1.0, 0.0)),
            ],
        },
        emissive_strength: 0.7,
        enabled: false,
        one_shot: false,
        ..Default::default()
    };
    let entity = spawn_entities(world, nightshade::ecs::PARTICLE_EMITTER, 1)[0];
    world.set(entity, emitter);
    entity
}

/// Draws the static control legend.
pub fn spawn_help(world: &mut World) {
    let lines = [
        "Drag the pool: ripples    Drag the ball: move it",
        "Drag outside the pool: orbit    Scroll: zoom",
        "Space: pause    Q or Esc: quit",
    ];
    let mut y = 232.0;
    for (index, line) in lines.iter().enumerate() {
        let font_size = if index == 0 { 26.0 } else { 18.0 };
        spawn_ui_text_with_properties(
            world,
            *line,
            Vec2::new(18.0, y),
            TextProperties {
                font_size,
                color: Vec4::new(0.95, 0.98, 1.0, 1.0),
                outline_width: 0.04,
                outline_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
                ..Default::default()
            },
        );
        y += font_size + 8.0;
    }
}

fn tile_material(tiling: f32) -> Material {
    let transform = TextureTransform {
        scale: [tiling, tiling],
        ..Default::default()
    };
    Material {
        base_color: [0.78, 0.8, 0.85, 1.0],
        base_texture: Some(TILE_TEXTURE.to_string()),
        base_texture_transform: transform,
        normal_texture: Some(TILE_NORMAL_TEXTURE.to_string()),
        normal_texture_transform: transform,
        normal_scale: 0.7,
        roughness: 0.35,
        metallic: 0.0,
        ..Default::default()
    }
}

fn generate_normal_map(bytes: &[u8]) -> (Vec<u8>, u32, u32) {
    let image = nightshade::prelude::image::load_from_memory(bytes)
        .expect("decode tile texture")
        .to_luma8();
    let (width, height) = image.dimensions();
    let strength = 2.5;
    let sample = |x: i32, y: i32| -> f32 {
        let sample_x = x.rem_euclid(width as i32) as u32;
        let sample_y = y.rem_euclid(height as i32) as u32;
        image.get_pixel(sample_x, sample_y)[0] as f32 / 255.0
    };
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let dx = sample(x as i32 + 1, y as i32) - sample(x as i32 - 1, y as i32);
            let dy = sample(x as i32, y as i32 + 1) - sample(x as i32, y as i32 - 1);
            let normal =
                nalgebra_glm::normalize(&nalgebra_glm::vec3(-dx * strength, -dy * strength, 1.0));
            let index = ((y * width + x) * 4) as usize;
            rgba[index] = ((normal.x * 0.5 + 0.5) * 255.0) as u8;
            rgba[index + 1] = ((normal.y * 0.5 + 0.5) * 255.0) as u8;
            rgba[index + 2] = ((normal.z * 0.5 + 0.5) * 255.0) as u8;
            rgba[index + 3] = 255;
        }
    }
    (rgba, width, height)
}
