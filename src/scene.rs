//! Scene setup: the camera, the control panel, the rain particles, and the help
//! overlay. The pool, the water, and the ball are all drawn by the GPU pass in
//! `water_pass.rs`, so nothing else lives in the ECS scene. Everything works in
//! the normalized pool space where the pool spans [-1, 1] in x and z.

use nightshade::prelude::*;

/// Half-width of the pool in the normalized space the pass renders.
pub const POOL_HALF: f32 = 1.0;
/// Rest height of the water surface.
pub const WATER_Y: f32 = 0.0;
/// Ball radius in the normalized space (matches the original demo).
pub const SPHERE_RADIUS: f32 = 0.25;

/// Spawns the orbit camera framing the pool and returns its entity. The
/// initialize system activates it through its `ActiveCamera` param.
pub fn spawn_camera(world: &mut World) -> Entity {
    nightshade::ecs::camera::commands::spawn_pan_orbit_camera_at(
        world,
        Vec3::new(0.0, -0.2, 0.0),
        4.0,
        3.7,
        0.42,
        "Camera".to_string(),
    )
}

/// Handles to the interactive widgets in the control panel.
pub struct Controls {
    pub rain: Entity,
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
            max: Vec2::new(216.0, 184.0),
        },
    );
    let content = widget::<UiPanelData>(tree.world_mut(), panel)
        .map(|data| data.content_entity)
        .unwrap_or(panel);
    let mut controls = Controls {
        rain: Entity::default(),
        ball: Entity::default(),
        reset: Entity::default(),
    };
    tree.in_parent(content, |tree| {
        controls.rain = tree.add_checkbox("Rain", false);
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
    let emitter = ParticleEmitter {
        emitter_type: EmitterType::Sparks,
        shape: EmitterShape::Box {
            half_extents: Vec3::new(POOL_HALF, 0.02, POOL_HALF),
        },
        position: Vec3::new(0.0, 1.6, 0.0),
        direction: Vec3::new(0.0, -1.0, 0.0),
        spawn_rate: 450.0,
        burst_count: 0,
        particle_lifetime_min: 0.5,
        particle_lifetime_max: 0.7,
        initial_velocity_min: 1.8,
        initial_velocity_max: 2.6,
        velocity_spread: 0.03,
        gravity: Vec3::new(0.0, -5.0, 0.0),
        drag: 0.0,
        size_start: 0.045,
        size_end: 0.02,
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
    let mut y = 200.0;
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
