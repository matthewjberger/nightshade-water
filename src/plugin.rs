//! The water demo plugin: sets up the camera and UI, registers the GPU water
//! pass, and feeds interaction (ripple drops, the draggable ball, rain, reset)
//! into it. Everything works in the normalized pool space where the pool spans
//! [-1, 1] in x and z, the floor is at y = -1, and the water rests at y = 0.

use crate::scene::{self, Controls, POOL_HALF, SPHERE_RADIUS};
use crate::water_pass::{WaterGpuPass, WaterParams};
use nightshade::ecs::camera::components::PanOrbitCamera;
use nightshade::prelude::*;
use nightshade::render::particles::ParticleEmitter;
use std::sync::{Arc, Mutex};

const DROP_RADIUS: f32 = 0.03;
const DROP_STRENGTH: f32 = 0.01;
const RAIN_DROP_RADIUS: f32 = 0.03;
const RAIN_DROP_STRENGTH: f32 = 0.008;
/// Seconds between scattered rain drops on the height field.
const RAIN_INTERVAL: f32 = 0.045;
const BALL_START: [f32; 3] = [-0.4, -0.75, 0.2];

#[derive(Clone, Copy, PartialEq)]
enum DragMode {
    None,
    Ball,
    Drops,
    Orbit,
}

/// A single ripple to inject into the height field, in texture UV space.
#[derive(Clone, Copy)]
struct DropSpec {
    center: Vec2,
    radius: f32,
    strength: f32,
}

/// App-wide state for the demo.
pub struct WaterState {
    params: Arc<Mutex<WaterParams>>,
    camera_entity: Entity,
    rain_emitter: Entity,
    controls: Controls,
    ball_center: Vec3,
    ball_previous_center: Vec3,
    ball_velocity: Vec3,
    ball_rotation: Quat,
    drag_mode: DragMode,
    drag_plane_normal: Vec3,
    drag_prev_hit: Vec3,
    pending_drop: Option<DropSpec>,
    paused: bool,
    rain_enabled: bool,
    ball_present: bool,
    reset_requested: bool,
    rain_accumulator: f32,
    rng: u32,
}

fn next_random(state: &mut u32) -> f32 {
    let mut value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    (value >> 8) as f32 / (1 << 24) as f32
}

/// The demo plugin.
pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        app.world.res_mut::<Window>().title = "Nightshade Water".to_string();
        let params = Arc::new(Mutex::new(WaterParams::default()));

        app.insert_resource(WaterState {
            params: params.clone(),
            camera_entity: Entity::default(),
            rain_emitter: Entity::default(),
            controls: Controls {
                rain: Entity::default(),
                ball: Entity::default(),
                reset: Entity::default(),
            },
            ball_center: Vec3::new(BALL_START[0], BALL_START[1], BALL_START[2]),
            ball_previous_center: Vec3::new(BALL_START[0], BALL_START[1], BALL_START[2]),
            ball_velocity: Vec3::zeros(),
            ball_rotation: Quat::identity(),
            drag_mode: DragMode::None,
            drag_plane_normal: Vec3::new(0.0, 0.0, 1.0),
            drag_prev_hit: Vec3::zeros(),
            pending_drop: None,
            paused: false,
            rain_enabled: false,
            ball_present: true,
            reset_requested: false,
            rain_accumulator: 0.0,
            rng: 0x9e37_79b9,
        });

        let graph_params = params;
        app.add_render_graph_config(move |graph, device, _format, resources| {
            let pass = WaterGpuPass::new(device, graph_params.clone());
            render_graph_add_pass(
                graph,
                Box::new(pass),
                &[("color", resources.scene_color), ("depth", resources.depth)],
            )
            .unwrap();
        });

        app.add_system(Stage::Startup, initialize);
        app.add_systems(Stage::Update, (handle_input, simulate, update_title));
    }
}

fn initialize(
    mut state: ResMut<WaterState>,
    mut render_settings: ResMut<RenderSettings>,
    mut debug_draw: ResMut<DebugDraw>,
    mut active_camera: ResMut<ActiveCamera>,
    world: &mut World,
) {
    render_settings.atmosphere = Atmosphere::Hdr;
    render_settings.show_sky = true;
    render_settings.bloom_enabled = false;
    debug_draw.show_grid = false;

    scene::load_environment(world);
    state.rain_emitter = scene::spawn_rain_emitter(world);
    let camera = scene::spawn_camera(world);
    active_camera.0 = Some(camera);
    state.camera_entity = camera;
    state.controls = scene::spawn_controls(world);
    scene::spawn_help(world);
}

fn handle_input(mut state: ResMut<WaterState>, input: Res<Input>, world: &mut World) {
    for event in ui_events(world).to_vec() {
        match event {
            UiEvent::CheckboxChanged { entity, value } if entity == state.controls.rain => {
                state.rain_enabled = value;
                set_emitter_enabled(world, state.rain_emitter, value);
            }
            UiEvent::CheckboxChanged { entity, value } if entity == state.controls.ball => {
                state.ball_present = value;
            }
            UiEvent::ButtonClicked(entity) if entity == state.controls.reset => {
                state.reset_requested = true;
            }
            _ => {}
        }
    }

    let mouse = input.mouse;
    let cursor = mouse.position;
    let left_pressed = mouse.state.contains(MouseState::LEFT_JUST_PRESSED);
    let left_held = mouse.state.contains(MouseState::LEFT_CLICKED);
    let cursor_moved = mouse.position_delta.norm() > 1.5;
    let toggle_pause = input.keyboard.just_pressed(KeyCode::Space);
    let quit = input.keyboard.just_pressed(KeyCode::KeyQ);

    if quit {
        world.res_mut::<Window>().should_exit = true;
    }
    if toggle_pause {
        state.paused = !state.paused;
    }

    if left_pressed {
        state.drag_mode = classify_press(&state, world, cursor);
        if state.drag_mode == DragMode::Ball {
            match grab_plane(&state, world, cursor) {
                Some((normal, hit)) => {
                    state.drag_plane_normal = normal;
                    state.drag_prev_hit = hit;
                }
                None => state.drag_mode = DragMode::Orbit,
            }
        }
    }
    if !left_held {
        state.drag_mode = DragMode::None;
    }

    let camera_enabled = !matches!(state.drag_mode, DragMode::Ball | DragMode::Drops);
    if let Some(camera) = world.get_mut::<PanOrbitCamera>(state.camera_entity) {
        camera.enabled = camera_enabled;
    }

    state.pending_drop = None;
    match state.drag_mode {
        DragMode::Ball => {
            if let Some(ray) = PickingRay::from_screen_position(world, cursor)
                && let Some(hit) = intersect_ray_plane(
                    ray.origin,
                    ray.direction,
                    state.drag_prev_hit,
                    state.drag_plane_normal,
                )
            {
                let delta = hit - state.drag_prev_hit;
                let mut center = state.ball_center + delta;
                center.x = center.x.clamp(SPHERE_RADIUS - 1.0, 1.0 - SPHERE_RADIUS);
                center.y = center.y.clamp(SPHERE_RADIUS - 1.0, 10.0);
                center.z = center.z.clamp(SPHERE_RADIUS - 1.0, 1.0 - SPHERE_RADIUS);
                state.ball_center = center;
                state.ball_velocity = Vec3::zeros();
                state.drag_prev_hit = hit;
            }
        }
        DragMode::Drops => {
            if (left_pressed || cursor_moved)
                && let Some(point) = get_ground_position_from_screen(world, cursor, scene::WATER_Y)
                && point.x.abs() <= POOL_HALF
                && point.z.abs() <= POOL_HALF
            {
                state.pending_drop = Some(DropSpec {
                    center: Vec2::new(point.x * 0.5 + 0.5, point.z * 0.5 + 0.5),
                    radius: DROP_RADIUS,
                    strength: DROP_STRENGTH,
                });
            }
        }
        DragMode::None | DragMode::Orbit => {}
    }
}

fn simulate(mut state: ResMut<WaterState>, time: Res<Time>) {
    let delta = time.delta_time.min(0.05);
    if !state.paused && state.ball_present && state.drag_mode != DragMode::Ball {
        step_ball_physics(&mut state, delta);
    }
    if !state.paused && state.ball_present {
        apply_ball_rotation(&mut state, delta);
    }

    if state.rain_enabled && !state.paused {
        state.rain_accumulator += delta;
        if state.rain_accumulator >= RAIN_INTERVAL && state.pending_drop.is_none() {
            state.rain_accumulator = 0.0;
            let center = Vec2::new(next_random(&mut state.rng), next_random(&mut state.rng));
            state.pending_drop = Some(DropSpec {
                center,
                radius: RAIN_DROP_RADIUS,
                strength: RAIN_DROP_STRENGTH,
            });
        }
    } else {
        state.rain_accumulator = 0.0;
    }

    let hidden = Vec3::new(0.0, 100.0, 0.0);
    let sphere_center = if state.ball_present {
        state.ball_center
    } else {
        hidden
    };
    let sphere_previous = if state.ball_present {
        state.ball_previous_center
    } else {
        hidden
    };

    let params_handle = state.params.clone();
    let mut params = params_handle.lock().unwrap();
    params.sphere_old = [sphere_previous.x, sphere_previous.y, sphere_previous.z];
    params.sphere_new = [sphere_center.x, sphere_center.y, sphere_center.z];
    params.sphere_radius = SPHERE_RADIUS;
    let rotation = nalgebra_glm::quat_to_mat3(&state.ball_rotation);
    let column = |index: usize| {
        [
            rotation.column(index)[0],
            rotation.column(index)[1],
            rotation.column(index)[2],
            0.0,
        ]
    };
    params.sphere_rotation = [column(0), column(1), column(2)];
    params.sphere_visible = state.ball_present;
    if state.reset_requested {
        params.reset = true;
    }
    match state.pending_drop.take() {
        Some(drop_spec) if !state.paused => {
            params.drop_active = true;
            params.drop_center = [drop_spec.center.x, drop_spec.center.y];
            params.drop_radius = drop_spec.radius;
            params.drop_strength = drop_spec.strength;
        }
        _ => params.drop_active = false,
    }
    drop(params);

    state.reset_requested = false;
    state.ball_previous_center = sphere_center;
}

fn update_title(state: Res<WaterState>, time: Res<Time>, mut window: ResMut<Window>) {
    let fps = time.frames_per_second;
    let status = if state.paused { "paused" } else { "running" };
    window.title = format!("Nightshade Water | {fps:.0} FPS | {status}");
}

fn set_emitter_enabled(world: &mut World, emitter: Entity, enabled: bool) {
    if let Some(component) = world.get_mut::<ParticleEmitter>(emitter) {
        component.enabled = enabled;
    }
}

fn classify_press(state: &WaterState, world: &World, cursor: Vec2) -> DragMode {
    if state.ball_present
        && let Some(ray) = PickingRay::from_screen_position(world, cursor)
        && ray_hits_sphere(ray.origin, ray.direction, state.ball_center, SPHERE_RADIUS)
    {
        return DragMode::Ball;
    }
    if let Some(point) = get_ground_position_from_screen(world, cursor, scene::WATER_Y)
        && point.x.abs() <= POOL_HALF
        && point.z.abs() <= POOL_HALF
    {
        return DragMode::Drops;
    }
    DragMode::Orbit
}

fn step_ball_physics(state: &mut WaterState, delta: f32) {
    let radius = SPHERE_RADIUS;
    let submerged = ((radius - state.ball_center.y) / (2.0 * radius)).clamp(0.0, 1.0);
    let gravity = Vec3::new(0.0, -4.0, 0.0);
    state.ball_velocity += gravity * (delta - 1.1 * delta * submerged);

    let speed_squared = state.ball_velocity.dot(&state.ball_velocity);
    if speed_squared > 1.0e-6 {
        let drag = state.ball_velocity.normalize() * (submerged * delta * speed_squared);
        state.ball_velocity -= drag;
    }

    state.ball_center += state.ball_velocity * delta;
    if state.ball_center.y < radius - 1.0 {
        state.ball_center.y = radius - 1.0;
        state.ball_velocity.y = state.ball_velocity.y.abs() * 0.7;
    }

    let limit = 1.0 - radius;
    state.ball_center.x = state.ball_center.x.clamp(-limit, limit);
    state.ball_center.z = state.ball_center.z.clamp(-limit, limit);
}

fn apply_ball_rotation(state: &mut WaterState, delta: f32) {
    let drift_axis = nalgebra_glm::normalize(&Vec3::new(0.25, 1.0, 0.15));
    state.ball_rotation =
        nalgebra_glm::quat_angle_axis(0.35 * delta, &drift_axis) * state.ball_rotation;

    let motion = state.ball_center - state.ball_previous_center;
    let horizontal = Vec3::new(motion.x, 0.0, motion.z);
    let distance = horizontal.norm();
    if distance > 1.0e-6 {
        let direction = horizontal / distance;
        let axis = nalgebra_glm::cross(&Vec3::new(0.0, 1.0, 0.0), &direction);
        let axis_length = axis.norm();
        if axis_length > 1.0e-6 {
            let angle = distance / SPHERE_RADIUS;
            let roll = nalgebra_glm::quat_angle_axis(angle, &(axis / axis_length));
            state.ball_rotation = roll * state.ball_rotation;
        }
    }
    state.ball_rotation = nalgebra_glm::quat_normalize(&state.ball_rotation);
}

fn grab_plane(state: &WaterState, world: &World, cursor: Vec2) -> Option<(Vec3, Vec3)> {
    let ray = PickingRay::from_screen_position(world, cursor)?;
    let normal = -ray.direction.normalize();
    let hit = intersect_ray_plane(ray.origin, ray.direction, state.ball_center, normal)?;
    Some((normal, hit))
}

fn intersect_ray_plane(
    origin: Vec3,
    direction: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Option<Vec3> {
    let denominator = plane_normal.dot(&direction);
    if denominator.abs() < 1.0e-6 {
        return None;
    }
    let distance = plane_normal.dot(&(plane_point - origin)) / denominator;
    if distance < 0.0 {
        return None;
    }
    Some(origin + direction * distance)
}

fn ray_hits_sphere(origin: Vec3, direction: Vec3, center: Vec3, radius: f32) -> bool {
    let to_center = origin - center;
    let a = direction.dot(&direction);
    let b = 2.0 * to_center.dot(&direction);
    let c = to_center.dot(&to_center) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return false;
    }
    (-b - discriminant.sqrt()) / (2.0 * a) > 0.0
}
